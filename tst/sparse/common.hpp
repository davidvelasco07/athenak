#ifndef TST_SPARSE_COMMON_HPP_
#define TST_SPARSE_COMMON_HPP_

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sparse/sparse_variables.hpp"

namespace sparse_test {

inline void Require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

class Arguments {
 public:
  Arguments(int argc, char **argv) {
    for (int i = 1; i < argc; i += 2) {
      Require(i+1 < argc && std::string(argv[i]).find("--") == 0,
              "Arguments must be --name value pairs");
      Require(values_.emplace(argv[i]+2, argv[i+1]).second, "Duplicate argument");
    }
  }
  std::string String(const std::string &key, const std::string &fallback) {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    const auto value = it->second;
    values_.erase(it);
    return value;
  }
  int Integer(const std::string &key, int fallback) {
    const auto raw = String(key, std::to_string(fallback));
    std::size_t end;
    const int result = std::stoi(raw, &end);
    Require(end == raw.size(), "Invalid integer: " + key);
    return result;
  }
  double Number(const std::string &key, double fallback) {
    const auto raw = String(key, std::to_string(fallback));
    std::size_t end;
    const double result = std::stod(raw, &end);
    Require(end == raw.size() && std::isfinite(result), "Invalid number: " + key);
    return result;
  }
  void Finish() const { Require(values_.empty(), "Unknown argument"); }

 private:
  std::map<std::string, std::string> values_;
};

inline sparse::Registry MakeRegistry(int count) {
  Require(count > 0 && count < 1000000, "Invalid field count");
  std::vector<sparse::Field> fields;
  for (int f = 0; f < count; ++f) {
    fields.push_back({17*f-5, "field_" + std::to_string(f), 0.0});
  }
  return sparse::Registry(std::move(fields));
}

inline void PrintEnvironment() {
  std::cout << std::setprecision(17)
            << "{\"type\":\"environment\",\"execution_space\":\""
            << DevExeSpace::name() << "\",\"real_bytes\":" << sizeof(Real) << "}\n";
}

using Range = Kokkos::RangePolicy<DevExeSpace, Kokkos::IndexType<std::int64_t>>;

}  // namespace sparse_test
#endif  // TST_SPARSE_COMMON_HPP_
