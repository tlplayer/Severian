#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { void *ptr; } MlirContext;
typedef struct { void *ptr; } MlirDialectRegistry;
typedef struct { const void *ptr; } MlirDialectHandle;
typedef struct { const void *ptr; } MlirModule;
typedef struct { void *ptr; } MlirOperation;
typedef struct { void *ptr; } MlirPassManager;
typedef struct { void *ptr; } MlirOpPassManager;
typedef struct { const char *data; size_t length; } MlirStringRef;
typedef struct { int8_t value; } MlirLogicalResult;
typedef void (*MlirStringCallback)(MlirStringRef, void *);

extern MlirDialectRegistry mlirDialectRegistryCreate(void);
extern void mlirDialectRegistryDestroy(MlirDialectRegistry);
extern void mlirRegisterAllDialects(MlirDialectRegistry);
extern MlirDialectHandle mlirGetDialectHandle__stablehlo__(void);
extern void mlirDialectHandleInsertDialect(MlirDialectHandle, MlirDialectRegistry);
extern MlirContext mlirContextCreateWithRegistry(MlirDialectRegistry, _Bool);
extern void mlirContextDestroy(MlirContext);
extern void mlirRegisterAllPasses(void);
extern void mlirRegisterAllStablehloPasses(void);
extern MlirModule mlirModuleCreateParse(MlirContext, MlirStringRef);
extern MlirOperation mlirModuleGetOperation(MlirModule);
extern void mlirModuleDestroy(MlirModule);
extern MlirPassManager mlirPassManagerCreate(MlirContext);
extern MlirOpPassManager mlirPassManagerGetAsOpPassManager(MlirPassManager);
extern MlirLogicalResult mlirParsePassPipeline(MlirOpPassManager, MlirStringRef, MlirStringCallback, void *);
extern MlirLogicalResult mlirPassManagerRunOnOp(MlirPassManager, MlirOperation);
extern void mlirPassManagerDestroy(MlirPassManager);
extern void mlirOperationPrint(MlirOperation, MlirStringCallback, void *);

static MlirStringRef string_ref(const char *data, size_t length) {
    MlirStringRef value = {data, length};
    return value;
}

static void write_chunk(MlirStringRef value, void *stream) {
    fwrite(value.data, 1, value.length, stream);
}

static char *read_input(size_t *length) {
    size_t capacity = 65536;
    char *input = malloc(capacity);
    if (input == NULL) return NULL;
    *length = 0;
    for (;;) {
        if (*length == capacity) {
            capacity *= 2;
            char *grown = realloc(input, capacity);
            if (grown == NULL) {
                free(input);
                return NULL;
            }
            input = grown;
        }
        size_t read = fread(input + *length, 1, capacity - *length, stdin);
        *length += read;
        if (read == 0) break;
    }
    return input;
}

int main(int argc, char **argv) {
    const char *pipeline = argc > 1 ? argv[1]
        : "builtin.module(stablehlo-legalize-to-linalg)";
    size_t source_length = 0;
    char *source = read_input(&source_length);
    if (source == NULL) return 2;

    MlirDialectRegistry registry = mlirDialectRegistryCreate();
    mlirRegisterAllDialects(registry);
    mlirDialectHandleInsertDialect(mlirGetDialectHandle__stablehlo__(), registry);
    MlirContext context = mlirContextCreateWithRegistry(registry, 0);
    mlirDialectRegistryDestroy(registry);
    mlirRegisterAllPasses();
    mlirRegisterAllStablehloPasses();

    MlirModule module = mlirModuleCreateParse(context, string_ref(source, source_length));
    free(source);
    if (module.ptr == NULL) {
        mlirContextDestroy(context);
        return 3;
    }
    MlirPassManager manager = mlirPassManagerCreate(context);
    MlirLogicalResult parsed = mlirParsePassPipeline(
        mlirPassManagerGetAsOpPassManager(manager),
        string_ref(pipeline, __builtin_strlen(pipeline)),
        write_chunk,
        stderr
    );
    if (!parsed.value || !mlirPassManagerRunOnOp(manager, mlirModuleGetOperation(module)).value) {
        mlirPassManagerDestroy(manager);
        mlirModuleDestroy(module);
        mlirContextDestroy(context);
        return 4;
    }
    mlirOperationPrint(mlirModuleGetOperation(module), write_chunk, stdout);
    fputc('\n', stdout);
    mlirPassManagerDestroy(manager);
    mlirModuleDestroy(module);
    mlirContextDestroy(context);
    return 0;
}
