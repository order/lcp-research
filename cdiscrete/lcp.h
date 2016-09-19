#ifndef __Z_LCP_INCLUDED__
#define __Z_LCP_INCLUDED__

struct LCP{
  LCP(const mat &,
      const vec &);
  mat M;
  vec q;
  void write(const string &);
};

vector<sp_mat> build_E_blocks(const Simulator * sim,
                              const TriMesh & mesh,
                              double gamma,
                              bool strip_oob);
sp_mat build_M(const vector<sp_mat> & E_blocks);

vec build_q(const Simulator * sim,
            const TriMesh & mesh,
            bool strip_oob);
LCP build_lcp(const Simulator * sim,
              const TriMesh & mesh,
              double gamma,
              bool strip_oob);
#endif
