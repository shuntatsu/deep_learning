#include <iostream>
#include <stdio.h>
using namespace std;

class List
{
public:
    int num;
    List *next = nullptr;
    List(int n){
        num = n;
    }
};

int main(void){
    List *obj1 = new List(1);
    List *obj2 = new List(2);
    List *obj3 = new List(3);
    List *obj4 = new List(4);

    obj1->next = obj2;
    obj2->next = obj3;
    obj3->next = obj4;

    List *obj = obj1;
    for(int i=0; i<4; i++)
    {
        cout << "num:" << obj->num << endl;
        obj = obj->next;
    }

    return 0;
}