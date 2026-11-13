#ifndef TOOLS__JSONL_RECORDER_HPP
#define TOOLS__JSONL_RECORDER_HPP

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace tools
{
// 主线程逐帧写入调试样本；写入失败后停用记录，不影响云台控制循环。
class JsonlRecorder
{
public:
  explicit JsonlRecorder(const std::string & path);
  bool write(const nlohmann::json & sample);
  bool enabled() const { return enabled_; }

private:
  std::ofstream output_;
  bool enabled_ = false;
};
}  // namespace tools

#endif  // TOOLS__JSONL_RECORDER_HPP
