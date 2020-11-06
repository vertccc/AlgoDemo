#include<iostream>

extern "C"{

int add_xy(int x, int y){
    return x + y;
}

int fib(int n){
    if(n <=2){
        return 1;
    }
    return fib(n-1) + fib(n-2);
}

void print_xy(){
    std::cout << "yes" << std::endl;
}

}


// g++ -fPIC -shared -o x.so x.cpp
// -o 输出
// -fPIC 作用于编译阶段，告诉编译器产生与位置无关代码 Position-Independent Code
//       这样一来，产生的代码中就没有绝对地址了，全部使用相对地址，
//       所以代码可以被加载器加载到内存的任意位置，都可以正确的执行。
//       这正是共享库所要求的，共享库被加载时，在内存的位置不是固定的。
// -shared 如果想创建一个动态链接库，-shared选项。输入文件可以是源文件、汇编文件或者目标文件。
