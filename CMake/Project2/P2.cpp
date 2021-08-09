#include "P2.h"

P2::P2(int b)
: P1(b)
{
    this->a = b;
}
int P2::get()
{
    return this->a;
}