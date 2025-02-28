(define (domain blocks)
  (:requirements :strips :typing:equality
                 :universal-preconditions
                 :conditional-effects)
  (:types physob)
  (:predicates   
      (ontable ?x - physob) ; x is in the table
      (clear ?x - physob)  ; there is nothing over x
      (on ?x ?y - physob))  ; x is over y
  
  (:action move  ; move x over y
             :parameters (?x ?y - physob)
             :precondition (and (clear ?x) (clear ?y)(not (= ?x ?y)))
             :effect (
              and (not (clear ?y)) (on ?x ?y) 
              (forall(?z)(when(on ?x ?z)(and (clear ?z)(not (on ?x ?z)))))
             )
  )

  (:action moveToTable
             :parameters (?x - physob)
             :precondition (and (clear ?x) (not (ontable ?x)))
             :effect (
              and (ontable ?x)
              (forall(?z)(when(on ?x ?z)(and (clear ?z)(not (on ?x ?z)))))
             )
  )
)