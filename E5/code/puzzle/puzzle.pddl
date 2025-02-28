(define (domain puzzle)
  (:requirements :strips :equality:typing)
  (:types num loc)
  (:constants
    B - num
  )
  (:predicates  (adjacent ?p1 ?p2 - loc)
                (at ?t - num ?p - loc))

  (:action slide
             :parameters (?t - num ?x ?y - loc)
             :precondition (and (at B ?y) (at ?t ?x) (adjacent ?x ?y))
             :effect (and (at B ?x) (at ?t ?y) (not (at ?t ?x) ) (not (at B ?y))) 
  )
)