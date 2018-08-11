#include "basestruct.h"

PRead::PRead(int col, int row, int invalid)
{
	this->col = col;
	this->row = row;
	this->invalid = invalid;
}
int PRead::cols()
{
	return this->col;
}
int PRead::rows()
{
	return this->row;
}
int PRead::invalidValue()
{
	return this->invalid;
}