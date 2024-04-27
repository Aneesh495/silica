#include "test_framework.hpp"

int main() {
    return silica::test::TestRegistry::instance().run_all();
}
