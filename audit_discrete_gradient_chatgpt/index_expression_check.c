#include <stdint.h>
#include <stdio.h>
int main(){volatile int32_t n=23172;int32_t m=2*n-2,i=m-1;int32_t index=i*m+i;printf("N=%d signed diagonal index=%d expected=%lld\n",n,index,(long long)i*m+i);n=32770;int32_t pairs=n-1,cols=2*pairs-1;size_t entries=(size_t)(pairs*cols);printf("N=%d full gradient entries=%zu expected=%lld\n",n,entries,(long long)pairs*cols);}
