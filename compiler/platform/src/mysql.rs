pub fn source() -> &'static str {
    r#"
#include <mysql/mysql.h>

void *__sev_mysql_connect(void *host_raw, void *user_raw, void *password_raw, void *database_raw, int64_t port) {
  MYSQL *connection = mysql_init(NULL);
  if (!connection) return sev_failure("could not initialize MariaDB client");
  mysql_options(connection, MYSQL_SET_CHARSET_NAME, "utf8mb4");
  if (!mysql_real_connect(connection, host_raw, user_raw, password_raw, database_raw,
                          (unsigned int)port, NULL, 0)) {
    const char *message = mysql_error(connection);
    char *copy = strcpy(sev_allocate(strlen(message) + 1), message);
    mysql_close(connection);
    return sev_failure(copy);
  }
  return __sev_variant_new("ok", connection);
}

void *__sev_mysql_query(void *connection_raw, void *sql_raw) {
  MYSQL *connection = connection_raw;
  if (!connection) return sev_failure("invalid MariaDB connection");
  const char *sql = sql_raw;
  if (mysql_real_query(connection, sql, (unsigned long)strlen(sql)) != 0)
    return sev_failure(mysql_error(connection));
  MYSQL_RES *result = mysql_store_result(connection);
  if (!result) {
    if (mysql_field_count(connection) == 0) {
      sev_collection *empty = __sev_collection_new(0);
      return __sev_variant_new("ok", __sev_box_collection(empty));
    }
    return sev_failure(mysql_error(connection));
  }
  unsigned int columns = mysql_num_fields(result);
  sev_collection *rows = __sev_collection_new(0);
  MYSQL_ROW row;
  while ((row = mysql_fetch_row(result)) != NULL) {
    unsigned long *lengths = mysql_fetch_lengths(result);
    sev_collection *values = __sev_collection_new(0);
    for (unsigned int column = 0; column < columns; ++column) {
      size_t length = row[column] ? lengths[column] : 0;
      char *value = sev_allocate(length + 1);
      if (row[column]) memcpy(value, row[column], length);
      value[length] = '\0';
      __sev_collection_push(values, __sev_box_string(value));
    }
    __sev_collection_push(rows, __sev_box_collection(values));
  }
  mysql_free_result(result);
  return __sev_variant_new("ok", __sev_box_collection(rows));
}

void *__sev_mysql_execute(void *connection_raw, void *sql_raw) {
  MYSQL *connection = connection_raw;
  if (!connection) return sev_failure("invalid MariaDB connection");
  const char *sql = sql_raw;
  if (mysql_real_query(connection, sql, (unsigned long)strlen(sql)) != 0)
    return sev_failure(mysql_error(connection));
  MYSQL_RES *result = mysql_store_result(connection);
  if (result) mysql_free_result(result);
  return __sev_variant_new("ok", __sev_box_i64((int64_t)mysql_affected_rows(connection)));
}

void *__sev_mysql_close(void *connection_raw) {
  MYSQL *connection = connection_raw;
  if (!connection) return sev_failure("invalid MariaDB connection");
  mysql_close(connection);
  return __sev_variant_new("ok", NULL);
}
"#
}
