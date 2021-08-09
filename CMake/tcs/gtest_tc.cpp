#include "P2.h"
#include <gtest/gtest.h>

TEST(HelloTest, BasicAssertions) 
{
    P2 p2(10);
    EXPECT_EQ(p2.get(), 10);
}