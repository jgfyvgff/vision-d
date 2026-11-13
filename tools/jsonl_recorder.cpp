#include "jsonl_recorder.hpp"

#include <filesystem>
#include <stdexcept>

#include "tools/logger.hpp"

namespace tools
{
JsonlRecorder::JsonlRecorder(const std::string & path)
{
  if (path.empty()) throw std::invalid_argument("JSONL path must not be empty");
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  output_.open(path, std::ios::out | std::ios::trunc);
  if (!output_) throw std::runtime_error("Failed to open JSONL: " + path);
  enabled_ = true;
}

bool JsonlRecorder::write(const nlohmann::json & sample)
{
  if (!enabled_) return false;
  output_ << sample.dump() << '\n';
  // 每帧刷新用户态缓冲，异常退出时尽量保留最近的分析数据。
  output_.flush();
  if (output_) return true;
  enabled_ = false;
  tools::logger()->error("[JSONL] Write failed; recording disabled.");
  return false;
}
}  // namespace tools
