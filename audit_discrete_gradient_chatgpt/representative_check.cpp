#include "host_helpers.hpp"
int main(){
 int N=14,R=2*N-1;
 std::vector<int32_t>a(R),b(R);int u=N;
 a[u]=0;b[u]=1;int clade=u++;
 for(int i=2;i<9;i++){a[u]=clade;b[u]=i;clade=u++;}
 int ab=u++;a[ab]=clade;b[ab]=9;
 int ef=u++;a[ef]=12;b[ef]=13;
 int def=u++;a[def]=11;b[def]=ef;
 int cdef=u++;a[cdef]=10;b[cdef]=def;
 int root=u++;a[root]=ab;b[root]=cdef;
 std::vector<int32_t>ord,par,count,reps;compute_tree_summaries_blind(a.data(),b.data(),N,ord,par,count,reps);
 printf("clade 0..8 representative leaves:");for(int i=0;i<8;i++)printf(" %d",reps[8*clade+i]);printf("\n");
 return 0;
}
