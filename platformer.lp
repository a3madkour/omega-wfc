
cell(0..10,0..8).


{ solid(X,Y)} :- cell(X,Y).

floor_below(X,Y) :- solid(X,Y+1).

reachable(0,8-1).
reachable(X,Y) :- reachable(X-1,Y), solid(X,Y+1), not solid(X,Y).
reachable(X,Y) :- reachable(X-2,Y),  solid(X,Y+1), not solid(X-1,Y), not solid(X-1,Y-1), not solid(X,Y).
reachable(X,Y) :- reachable(X-1,Y+1), solid(X,Y+1), not solid(X,Y), not solid(X-1,Y). 


playable :- reachable(10-1,0), not solid(0,0), not solid(0,8-1).

:- not playable.

#show reachable/2.
#show solid/2.

