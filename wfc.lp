1 { assign(C,T):tile(T)} 1 :- cell(C).

edge_assign(A,C1,C2,T1,T2) :- adj(A,C1,C2), assign(C1,T1), assign(C2,T2).
pair_ok(A,T1,T2) :- pair_ppt(A,T1,T2,PPT).
:- edge_assign(A,C1,C2,T1,T2), not pair_ok(A,T1,T2).
 
n(H*W) :- shape((H,W)).
 
#const tile_tau = 80000.
%:- tile_ppt(T,PPT), n(N), not N*PPT - tile_tau <= #sum { 1000,C:assign(C,T) } <= N*PPT + tile_tau.
 
#const edge_tau = 800000.
% N is wrong below because we are doing 2x1 patterns, fewer sites
%:- pair_ppt(A,T1,P2,PPT), n(N), not N*PPT - edge_tau <= #sum { 1000,C1,C2:edge_assign(A,C1,C2,T1,T2) } <= N*PPT + edge_tau.
 
 
#show shape/1.
#show assign/2.
