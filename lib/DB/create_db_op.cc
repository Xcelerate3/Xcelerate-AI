#include "xcelerate/db/create_db_op.h"

namespace xcelerate {
REGISTER_CPU_OPERATOR(CreateDB, CreateDBOp<CPUContext>);

OPERATOR_SCHEMA(CreateDB).NumInputs(0).NumOutputs(1);

NO_GRADIENT(CreateDB);
} // namespace xcelerate
