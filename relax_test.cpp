#include <gurobi_c++.h>
#include <chrono>
#include <iostream>

static double now_sec(){using namespace std::chrono;return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char** argv){
    if(argc<2){std::cerr<<"usage"<<std::endl;return 1;}
    GRBEnv env(true);
    env.set(GRB_IntParam_LogToConsole,0);
    env.start();
    GRBModel model(env,argv[1]);
    model.set(GRB_IntParam_OutputFlag,0);
    double t0=now_sec();
    GRBModel r=model.relax();
    r.set(GRB_IntParam_OutputFlag,0);
    // report some parameters
    int meth = r.get(GRB_IntParam_Method);
    int thr = r.get(GRB_IntParam_Threads);
    std::cout<<"method="<<meth<<" threads="<<thr<<"\n";
    double t1=now_sec();
    std::cout<<"after_relax_build="<<(t1-t0)<<"\n";
    r.optimize();
    double t2=now_sec();
    std::cout<<"relax_opt_time="<<(t2-t1)<<"\n";
    return 0;
}
