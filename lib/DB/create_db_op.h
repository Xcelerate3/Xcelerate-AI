#ifndef xcelerate_DB_CREATE_DB_OP_H_
#define xcelerate_DB_CREATE_DB_OP_H_

#include "xcelerate/core/context.h"
#include "xcelerate/core/db.h"
#include "xcelerate/core/operator.h"

namespace xcelerate {

template <class Context>
class CreateDBOp final : public Operator<Context> {
 public:
  CreateDBOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        db_type_(OperatorBase::template GetSingleArgument<string>(
            "db_type",
            "leveldb")),
        db_name_(OperatorBase::template GetSingleArgument<string>("db", "")),
        num_shards_(
            OperatorBase::template GetSingleArgument<int>("num_shards", 1)),
        shard_id_(
            OperatorBase::template GetSingleArgument<int>("shard_id", 0)) {
    CAFFE_ENFORCE_GT(db_name_.size(), 0, "Must specify a db name.");
  }

  bool RunOnDevice() final {
    OperatorBase::Output<db::DBReader>(0)->Open(
        db_type_, db_name_, num_shards_, shard_id_);
    return true;
  }

 private:
  string db_type_;
  string db_name_;
  uint32_t num_shards_;
  uint32_t shard_id_;
  C10_DISABLE_COPY_AND_ASSIGN(CreateDBOp);
};

} // namespace xcelerate

#endif // xcelerate_DB_CREATE_DB_OP_H_
