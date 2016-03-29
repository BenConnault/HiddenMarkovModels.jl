type Atom{Label} end

A1=Atom{1}
A2=Atom{2}

a1=A1()
a2=A2()


@test instantiate(A1) == a1
@test instantiate(Tuple{A1}) == (a1,)
@test instantiate(typeof((a1,a2))) == (a1,a2)
@test instantiate(typeof((a1,a2,a2))) == (a1,a2,a2)
@test instantiate(typeof((a1,(a2,a1),a2,a2))) == (a1,(a2,a1),a2,a2)
