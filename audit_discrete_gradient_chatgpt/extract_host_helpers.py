from pathlib import Path
s=Path('snapshot/bench_rnasim_gpu.cu').read_text()
names=['relabel_from_rooted_tree','decode_cf_to_rooted','build_skeleton_metadata','compute_tree_summaries_blind','extract_active_skeleton','build_component_waves','compute_outgroup_rep8']
parts=['#include <vector>\n#include <queue>\n#include <algorithm>\n#include <cstdint>\n#include <cstring>\n#include <cstdio>\n']
for name in names:
 start=s.index('static void '+name+'(')
 brace=s.index('{',start); depth=1;i=brace+1
 while depth:
  depth+=(s[i]=='{')-(s[i]=='}');i+=1
 parts.append(s[start:i]+'\n')
Path('host_helpers.hpp').write_text('\n'.join(parts))
