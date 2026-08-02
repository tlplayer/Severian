pub(crate) fn source() -> &'static str {
    r#"
#include <sqlite3.h>

typedef struct { sqlite3 *connection; } sev_database;
typedef struct { int listener; uint16_t port; sqlite3 *connection; pthread_t thread; } sev_database_server;
typedef struct { uint16_t port; } sev_database_client;

static char *sev_database_copy(const char *text) {
  size_t size = strlen(text);
  char *copy = sev_allocate(size + 1);
  memcpy(copy, text, size + 1);
  return copy;
}

static sev_collection *sev_database_query_rows(sqlite3 *connection, const char *sql, char **error) {
  sqlite3_stmt *statement = NULL;
  int status = sqlite3_prepare_v2(connection, sql, -1, &statement, NULL);
  if (status != SQLITE_OK) {
    *error = sev_database_copy(sqlite3_errmsg(connection));
    return NULL;
  }
  sev_collection *rows = __sev_collection_new(0);
  int columns = sqlite3_column_count(statement);
  while ((status = sqlite3_step(statement)) == SQLITE_ROW) {
    sev_collection *row = __sev_collection_new(0);
    for (int column = 0; column < columns; ++column) {
      const unsigned char *value = sqlite3_column_text(statement, column);
      const char *text = value ? (const char *)value : "null";
      __sev_collection_push(row, __sev_box_string(sev_database_copy(text)));
    }
    __sev_collection_push(rows, __sev_box_collection(row));
  }
  if (status != SQLITE_DONE) {
    *error = sev_database_copy(sqlite3_errmsg(connection));
    sqlite3_finalize(statement);
    return NULL;
  }
  sqlite3_finalize(statement);
  return rows;
}

void *__sev_database_open(void *path_raw) {
  sev_database *database = sev_allocate(sizeof(*database));
  if (sqlite3_open((const char *)path_raw, &database->connection) != SQLITE_OK) {
    const char *message = sqlite3_errmsg(database->connection);
    sqlite3_close(database->connection);
    return sev_failure(message);
  }
  sqlite3_busy_timeout(database->connection, 5000);
  return __sev_variant_new("ok", database);
}

void *__sev_database_execute(void *database_raw, void *sql_raw) {
  sev_database *database = database_raw;
  char *message = NULL;
  if (!database || !database->connection) return sev_failure("database is closed");
  if (sqlite3_exec(database->connection, (const char *)sql_raw, NULL, NULL, &message) != SQLITE_OK) {
    void *failure = sev_failure(message ? message : sqlite3_errmsg(database->connection));
    sqlite3_free(message);
    return failure;
  }
  return __sev_variant_new("ok", __sev_box_i64(sqlite3_changes(database->connection)));
}

void *__sev_database_query(void *database_raw, void *sql_raw) {
  sev_database *database = database_raw;
  char *message = NULL;
  if (!database || !database->connection) return sev_failure("database is closed");
  sev_collection *rows = sev_database_query_rows(database->connection, (const char *)sql_raw, &message);
  if (!rows) return sev_failure(message ? message : "database query failed");
  return __sev_variant_new("ok", __sev_box_collection(rows));
}

void *__sev_database_close(void *database_raw) {
  sev_database *database = database_raw;
  if (!database || !database->connection) return sev_failure("database is already closed");
  if (sqlite3_close(database->connection) != SQLITE_OK) return sev_failure(sqlite3_errmsg(database->connection));
  database->connection = NULL;
  return __sev_variant_new("ok", NULL);
}

static bool sev_database_write_i64(int socket, int64_t value) {
  return sev_socket_write_all(socket, (const char *)&value, sizeof(value));
}

static bool sev_database_read_i64(int socket, int64_t *value) {
  return sev_socket_read_all(socket, (char *)value, sizeof(*value));
}

static bool sev_database_send_error(int socket, const char *message) {
  uint8_t status = 0;
  int64_t size = (int64_t)strlen(message);
  return sev_socket_write_all(socket, (const char *)&status, 1) &&
         sev_database_write_i64(socket, size) &&
         sev_socket_write_all(socket, message, (size_t)size);
}

static void *sev_database_server_worker(void *raw) {
  sev_database_server *server = raw;
  bool running = true;
  while (running) {
    int peer = accept(server->listener, NULL, NULL);
    if (peer < 0) continue;
    uint8_t operation = 0;
    int64_t sql_size = 0;
    if (!sev_socket_read_all(peer, (char *)&operation, 1)) { close(peer); continue; }
    if (operation == 'P') {
      uint8_t status = 1;
      sev_socket_write_all(peer, (const char *)&status, 1);
      close(peer);
      continue;
    }
    if (operation == 'S') {
      uint8_t status = 1;
      sev_socket_write_all(peer, (const char *)&status, 1);
      close(peer);
      running = false;
      continue;
    }
    if (!sev_database_read_i64(peer, &sql_size) || sql_size < 0 || sql_size > 16 * 1024 * 1024) {
      sev_database_send_error(peer, "invalid SQL request size");
      close(peer);
      continue;
    }
    char *sql = sev_allocate((size_t)sql_size + 1);
    if (!sev_socket_read_all(peer, sql, (size_t)sql_size)) { close(peer); continue; }
    sql[sql_size] = '\0';
    if (operation == 'E') {
      char *message = NULL;
      if (sqlite3_exec(server->connection, sql, NULL, NULL, &message) != SQLITE_OK) {
        sev_database_send_error(peer, message ? message : sqlite3_errmsg(server->connection));
        sqlite3_free(message);
      } else {
        uint8_t status = 1;
        sev_socket_write_all(peer, (const char *)&status, 1);
        sev_database_write_i64(peer, sqlite3_changes(server->connection));
      }
    } else if (operation == 'Q') {
      char *message = NULL;
      sev_collection *rows = sev_database_query_rows(server->connection, sql, &message);
      if (!rows) {
        sev_database_send_error(peer, message ? message : "database query failed");
      } else {
        uint8_t status = 1;
        sev_socket_write_all(peer, (const char *)&status, 1);
        sev_database_write_i64(peer, rows->size);
        for (int64_t row_index = 0; row_index < rows->size; ++row_index) {
          sev_value *boxed_row = rows->items[row_index];
          sev_collection *row = boxed_row->as.pointer;
          sev_database_write_i64(peer, row->size);
          for (int64_t column = 0; column < row->size; ++column) {
            sev_value *cell = row->items[column];
            int64_t size = (int64_t)strlen(cell->as.string);
            sev_database_write_i64(peer, size);
            sev_socket_write_all(peer, cell->as.string, (size_t)size);
          }
        }
      }
    } else {
      sev_database_send_error(peer, "unknown database operation");
    }
    free(sql);
    close(peer);
  }
  return NULL;
}

static int sev_database_connect_port(uint16_t port) {
  int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) return -1;
  struct sockaddr_in endpoint = {0};
  endpoint.sin_family = AF_INET;
  endpoint.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  endpoint.sin_port = htons(port);
  if (connect(socket_fd, (struct sockaddr *)&endpoint, sizeof(endpoint)) != 0) {
    close(socket_fd);
    return -1;
  }
  return socket_fd;
}

void *__sev_database_server_start(void *path_raw) {
  sev_database_server *server = sev_allocate(sizeof(*server));
  if (sqlite3_open((const char *)path_raw, &server->connection) != SQLITE_OK) {
    const char *message = sqlite3_errmsg(server->connection);
    sqlite3_close(server->connection);
    return sev_failure(message);
  }
  sqlite3_busy_timeout(server->connection, 5000);
  server->listener = socket(AF_INET, SOCK_STREAM, 0);
  if (server->listener < 0) { sqlite3_close(server->connection); return sev_failure("could not create database server socket"); }
  struct sockaddr_in endpoint = {0};
  endpoint.sin_family = AF_INET;
  endpoint.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(server->listener, (struct sockaddr *)&endpoint, sizeof(endpoint)) != 0 || syscall(SYS_listen, server->listener, 16) != 0) {
    close(server->listener);
    sqlite3_close(server->connection);
    return sev_failure("could not bind database server");
  }
  socklen_t endpoint_size = sizeof(endpoint);
  if (getsockname(server->listener, (struct sockaddr *)&endpoint, &endpoint_size) != 0) {
    close(server->listener);
    sqlite3_close(server->connection);
    return sev_failure("could not inspect database server address");
  }
  server->port = ntohs(endpoint.sin_port);
  if (pthread_create(&server->thread, NULL, sev_database_server_worker, server) != 0) {
    close(server->listener);
    sqlite3_close(server->connection);
    return sev_failure("could not start database server thread");
  }
  return __sev_variant_new("ok", server);
}

void *__sev_database_server_address(void *server_raw) {
  sev_database_server *server = server_raw;
  char *address = sev_allocate(32);
  snprintf(address, 32, "127.0.0.1:%u", server->port);
  return address;
}

void *__sev_database_server_connect(void *address_raw) {
  const char *address = address_raw;
  const char *colon = strrchr(address, ':');
  if (!colon) return sev_failure("database server address requires a port");
  long port = strtol(colon + 1, NULL, 10);
  if (port < 1 || port > 65535) return sev_failure("invalid database server port");
  int socket_fd = sev_database_connect_port((uint16_t)port);
  if (socket_fd < 0) return sev_failure("could not connect to database server");
  uint8_t operation = 'P', status = 0;
  bool success = sev_socket_write_all(socket_fd, (const char *)&operation, 1) &&
                 sev_socket_read_all(socket_fd, (char *)&status, 1) && status == 1;
  close(socket_fd);
  if (!success) return sev_failure("database server did not answer");
  sev_database_client *client = sev_allocate(sizeof(*client));
  client->port = (uint16_t)port;
  return __sev_variant_new("ok", client);
}

static void *sev_database_remote_error(int socket) {
  int64_t size = 0;
  if (!sev_database_read_i64(socket, &size) || size < 0 || size > 1024 * 1024) return sev_failure("invalid database server error");
  char *message = sev_allocate((size_t)size + 1);
  if (!sev_socket_read_all(socket, message, (size_t)size)) return sev_failure("incomplete database server error");
  message[size] = '\0';
  return sev_failure(message);
}

void *__sev_database_server_execute(void *client_raw, void *sql_raw) {
  sev_database_client *client = client_raw;
  int socket_fd = sev_database_connect_port(client->port);
  if (socket_fd < 0) return sev_failure("database server connection failed");
  uint8_t operation = 'E', status = 0;
  int64_t size = (int64_t)strlen((const char *)sql_raw), changes = 0;
  bool sent = sev_socket_write_all(socket_fd, (const char *)&operation, 1) &&
              sev_database_write_i64(socket_fd, size) &&
              sev_socket_write_all(socket_fd, (const char *)sql_raw, (size_t)size) &&
              sev_socket_read_all(socket_fd, (char *)&status, 1);
  if (!sent) { close(socket_fd); return sev_failure("database server request failed"); }
  if (!status) { void *failure = sev_database_remote_error(socket_fd); close(socket_fd); return failure; }
  if (!sev_database_read_i64(socket_fd, &changes)) { close(socket_fd); return sev_failure("database server response failed"); }
  close(socket_fd);
  return __sev_variant_new("ok", __sev_box_i64(changes));
}

void *__sev_database_server_query(void *client_raw, void *sql_raw) {
  sev_database_client *client = client_raw;
  int socket_fd = sev_database_connect_port(client->port);
  if (socket_fd < 0) return sev_failure("database server connection failed");
  uint8_t operation = 'Q', status = 0;
  int64_t size = (int64_t)strlen((const char *)sql_raw);
  bool sent = sev_socket_write_all(socket_fd, (const char *)&operation, 1) &&
              sev_database_write_i64(socket_fd, size) &&
              sev_socket_write_all(socket_fd, (const char *)sql_raw, (size_t)size) &&
              sev_socket_read_all(socket_fd, (char *)&status, 1);
  if (!sent) { close(socket_fd); return sev_failure("database server request failed"); }
  if (!status) { void *failure = sev_database_remote_error(socket_fd); close(socket_fd); return failure; }
  int64_t row_count = 0;
  if (!sev_database_read_i64(socket_fd, &row_count) || row_count < 0) { close(socket_fd); return sev_failure("invalid database row count"); }
  sev_collection *rows = __sev_collection_new(0);
  for (int64_t row_index = 0; row_index < row_count; ++row_index) {
    int64_t columns = 0;
    if (!sev_database_read_i64(socket_fd, &columns) || columns < 0) { close(socket_fd); return sev_failure("invalid database column count"); }
    sev_collection *row = __sev_collection_new(0);
    for (int64_t column = 0; column < columns; ++column) {
      int64_t cell_size = 0;
      if (!sev_database_read_i64(socket_fd, &cell_size) || cell_size < 0 || cell_size > 16 * 1024 * 1024) { close(socket_fd); return sev_failure("invalid database cell size"); }
      char *cell = sev_allocate((size_t)cell_size + 1);
      if (!sev_socket_read_all(socket_fd, cell, (size_t)cell_size)) { close(socket_fd); return sev_failure("incomplete database cell"); }
      cell[cell_size] = '\0';
      __sev_collection_push(row, __sev_box_string(cell));
    }
    __sev_collection_push(rows, __sev_box_collection(row));
  }
  close(socket_fd);
  return __sev_variant_new("ok", __sev_box_collection(rows));
}

void *__sev_database_server_close(void *client_raw) {
  (void)client_raw;
  return __sev_variant_new("ok", NULL);
}

void *__sev_database_server_stop(void *server_raw) {
  sev_database_server *server = server_raw;
  int socket_fd = sev_database_connect_port(server->port);
  if (socket_fd < 0) return sev_failure("could not stop database server");
  uint8_t operation = 'S', status = 0;
  bool stopped = sev_socket_write_all(socket_fd, (const char *)&operation, 1) &&
                 sev_socket_read_all(socket_fd, (char *)&status, 1) && status == 1;
  close(socket_fd);
  if (!stopped) return sev_failure("database server stop failed");
  pthread_join(server->thread, NULL);
  close(server->listener);
  sqlite3_close(server->connection);
  return __sev_variant_new("ok", NULL);
}
"#
}
