#include <iostream>
#include <stdio.h>
#include <vector>
// shared_ptr用:参照カウント方式
#include <memory>

using namespace std;

class Function;

// 変数を表現するためのクラス
class Variable
{
public:
    float a = 0;
    Function *creator = nullptr;

    // コンストラクタ 
    Variable(float a)
    {
        // this: 現在のオブジェクトを指すポインタ
        // メンバ変数を初期化
        this->a = a;
    }
};

using PVariable = shared_ptr<Variable>;

class Function
{
public:
    // 内部のメモリ領域を拡張し、新しい要素を格納するためのスペースを確保
    vector<PVariable> v;

    PVariable forward(PVariable v1, PVariable v2)
    {
        v.push_back(v1);
        v.push_back(v2);

        PVariable pv = PVariable(new Variable(0));
        pv->creator = this;

        pv->a += v1->a;
        pv->a += v2->a;
        return pv;
    }

    PVariable forward(PVariable v1)
    {
        v.push_back(v1);

        PVariable pv = PVariable(new Variable(0));
        pv->creator = this;

        pv->a += v1->a;
        return pv;
    }
};

void traverse(PVariable v)
{
    cout << v->a << endl;

    Function *f = v->creator;
    if(f == NULL)
    {
        return;
    }

    for(int i=0; i < f->v.size(); i++)
    {
        traverse(f->v[i]);
        cout << "size:" << f->v.size() << endl;
    }
}

int main(void)
{
    PVariable v1 = PVariable(new Variable(1));
    PVariable v2 = PVariable(new Variable(2));
    Function *f1 = new Function();
    Function *f2 = new Function();
    Function *f3 = new Function();

    PVariable r1 = f1->forward(v1, v2);
    PVariable r2 = f2->forward(r1);
    PVariable r3 = f3->forward(r2);

    traverse(r3);
    return 0;
}

/*
r3 (3)
│
└── f3
    │
    └── r2 (3)
        │
        └── f2
            │
            └── r1 (3)
                │
                └── f1
                    │
                    ├── v1 (1)
                size: 2   
                    │
                    └── v2 (2)
                size: 2       
        size: 1
size: 1

v1, v2はFunctionでないため、繰り返さない。
*/