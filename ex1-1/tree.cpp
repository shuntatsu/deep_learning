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
    string name;

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
    cout << "Variable " << v->name << " with value: " << v->a << " (use_count: " << v.use_count() << ")" << endl;

    Function *f = v->creator;
    if(f == NULL)
    {
        return;
    }

    for(int i=0; i < f->v.size(); i++)
    {
        traverse(f->v[i]);
        cout << "Variable :" << v->name << " ,size:" << f->v.size() << endl;
    }
}

int main(void)
{
    PVariable v1 = PVariable(new Variable(1));
    v1->name = "v1";
    PVariable v2 = PVariable(new Variable(2));
    v2->name = "v2";
    //v1とv2の参照カウントはそれぞれ1です。
    cout << "v1 use_count: " << v1.use_count() << endl;
    Function *f1 = new Function();
    Function *f2 = new Function();
    Function *f3 = new Function();

    PVariable r1 = f1->forward(v1, v2);
    r1->name = "r1";
    PVariable r2 = f2->forward(r1);
    r2->name = "r2";
    PVariable r3 = f3->forward(r2);
    r3->name = "r3";

    traverse(r3);

    // 参照カウントとサイズの確認
    cout << "After traverse:" << endl;
    cout << "v1 use_count: " << v1.use_count() << endl;
    cout << "v2 use_count: " << v2.use_count() << endl;
    cout << "r1 use_count: " << r1.use_count() << endl;
    cout << "r2 use_count: " << r2.use_count() << endl;
    cout << "r3 use_count: " << r3.use_count() << endl;
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

f->v.size()
f->v.size()は、Functionクラスのメンバ変数vのサイズを返します。
vはvector<PVariable>型で、Functionオブジェクトが保持しているPVariableの数を示します。

実行結果
Variable r3 with value: 3 (use_count: 2)
Variable r2 with value: 3 (use_count: 3)
Variable r1 with value: 3 (use_count: 3)
Variable v1 with value: 1 (use_count: 3)
Variable :r1 ,size:2
Variable v2 with value: 2 (use_count: 3)
Variable :r1 ,size:2
Variable :r2 ,size:1
Variable :r3 ,size:1
After traverse:
v1 use_count: 2
v2 use_count: 2
r1 use_count: 2
r2 use_count: 2
r3 use_count: 1
f1->v.size(): 2
f2->v.size(): 1
f3->v.size(): 1

*/


/*
循環参照の問題
このコードでは、FunctionオブジェクトがVariableオブジェクトをshared_ptrで保持し、VariableオブジェクトがFunctionオブジェクトを指すポインタを持っています。
このように、オブジェクトが互いに参照し合うと、参照カウントが0にならず、メモリが解放されません。
 */
