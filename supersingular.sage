"""
#*****************************************************************************
# Copyright (C) # Copyright (C) 2024 -- 2025
# Marco Streng <marco.streng@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# See license.txt
#*****************************************************************************

In this document we do verifications and sanity-checks for some of the
statements in *Explicit supersingular cyclic curves*.
We do this for each of the three sections 2, 3, and 4.


The results in **Section 2** (about M(6)) do not depend on computer verification.
Even the verification that the curve has good reduction outside of 3 is easy
to do by hand. We do the following sanity checks.

Check that the discriminant of f is consistent with bad reduction only at 3::

    sage: load("supersingular.sage")
    sage: [(m, f)] = curvesM6
    sage: m, f
    (3, x^4 - x)
    sage: f.discriminant().factor()
    -1 * 3^3

Check that it is supersingular ('ss') at the primes 2 and 5 to which Theorem 2.1 applies::

    sage: for p in [2, 5]:
    ....:     print(p % 3, p, newton_polygon(m, f, p)[2])
    2 2 ss
    2 5 ss

Check that the slopes of the Newton polygon at 7 (to which Theorem 2.1 does not apply)
are not all equal to 1/2::

    sage: p = 7
    sage: print(p % 3, p, newton_polygon(m, f, p)[2])
    1 7 [1/3, 1/3, 1/3, 2/3, 2/3, 2/3]


We go on with **Section 3** and the family M(8). We do this curve by curve.

We verify that the first curve has good reduction outside 2 and 3::

    sage: (m, f) = curvesM8[0]
    sage: m, f
    (2, x^7 + 6*x^5 + 9*x^3 + x)
    sage: f.discriminant().factor(), m.factor()
    (-1 * 2^6 * 3^8, 2)

And then we do a sanity check that the theorem holds for p = 7 and does not hold for p = 5::

    sage: p = 7
    sage: print(p % 4, p, newton_polygon(m, f, p)[2])
    3 7 ss

    sage: p = 5
    sage: print(p % 4, p, newton_polygon(m, f, p)[2])
    1 5 [1/3, 1/3, 1/3, 2/3, 2/3, 2/3]

We verify that the second curve has good reduction at 3 (and in fact everywhere
outside 2 and 7)::

    sage: (m, f) = curvesM8[1]
    sage: f.discriminant().factor(), m.factor()
    (-1 * 2^6 * 7^7, 2)

We verify that this curve is supersingular modulo 3::

    sage: p = 3
    sage: print(p % 4, p, newton_polygon(m, f, p)[2])
    3 3 ss

We do a sanity check that it is not supersingular modulo 5::

    sage: p = 5
    sage: print(p % 4, p, newton_polygon(m, f, p)[2])
    1 5 [1/3, 1/3, 1/3, 2/3, 2/3, 2/3]


Finally we do **Section 4** and the family M(16). Again we do this curve by curve.

For the first curve we verify that it has good reduction outside 3 and 5.
We also do a sanity check for all primes under 18 except for 3 and 5::

    sage: (m, f) = curvesM16[0]
    sage: (m, f)
    (5, x^4 - 24*x^3 + 3*x^2 + x)
    sage: f.discriminant().factor(), m.factor()
    (3^10, 5)

    sage: P234 = [p for p in prime_range(18) if p % 5 in [2, 3, 4]]
    sage: for p in P234:
    ....:     if p != 3:
    ....:         print(p % 5, p, newton_polygon(m, f, p)[2])
    2 2 ss
    2 7 ss
    3 13 ss
    2 17 ss

    sage: Pother = [p for p in prime_range(18) if p % 5 == 1]
    sage: for p in Pother:
    ....:     print(p % 5, p, newton_polygon(m, f, p)[2])
    1 11 [0, 0, 0, 1/3, 1/3, 1/3, 2/3, 2/3, 2/3, 1, 1, 1]

For the second curve we verify that it has supersingular good reduction at 3,
and we do the same sanity checks as well::

    sage: (m, f) = curvesM16[1]
    sage: (m, f)
    (5, x^4 - 7*x^2 + 7*x)
    sage: f.discriminant().factor(), m.factor()
    (7^4, 5)

    sage: for p in P234:
    ....:     if p != 7:
    ....:         print(p % 5, p, newton_polygon(m, f, p)[2])
    2 2 ss
    3 3 ss
    3 13 ss
    2 17 ss

    sage: for p in Pother:
    ....:     print(p % 5, p, newton_polygon(m, f, p)[2])
    1 11 [0, 0, 0, 1/3, 1/3, 1/3, 2/3, 2/3, 2/3, 1, 1, 1]
    

Finally as another sanity check we take a somewhat random curve
and see whether it is supersingular::

    sage: (m, f) = curvesM16[0]
    sage: f = f + 1
    sage: f.discriminant().factor()
    -1 * 373 * 25307
    
    sage: for p in [2,3,7,13,17]: # we skip 11 as it is computationally harder
    ....:     print(p%5, p, newton_polygon(m, f, p)[2])
    ....: 
    2 2 ss
    3 3 ss
    2 7 [1/4, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4, 3/4]
    3 13 [1/4, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4, 3/4]
    2 17 [1/4, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4, 3/4]
    
We only get supersingular reduction (by chance?) for the smaller primes.




"""

P.<x> = ZZ[]

curvesM6 = [(3, x^4 - x)]
curvesM8 = [(2, x*(x^6 + 6*x^4 + 9*x^2 + 1)), (2, x*(x^6+7*x^4+14*x^2+7))]
curvesM16 = [(5, x^4 - 24*x^3 + 3*x^2 + x), (5, x^4 - 7*x^2 + 7*x)]


def count_points_plane(m, f, q):
    """
    Gives the number of projective points of the curve y^m = f(x) in P^2(F_q).
    """
    d = f.degree()
    F = GF(q)
    
    if not d in [m-1, m, m+1]:
        raise NotImplementedError("Only implemented for |deg(f)-m| <= 1.")
    
    if gcd(q-1, m) == 1:
        # Every element of F_q has a unique mth root, giving q affine points.
        # There is also a unique point at infinity:
        # - if d == m + 1, then
        #   Y^m*Z = f(X/Z)*Z^(m+1) specializes to 0 = lc*X^(m+1),
        #   yielding exactly [0:1:0]
        # - if d \in {m-1, m}, then
        #   Y^m = f(X/Z)*Z^m specializes to Y^m = lc*X^m,
        #   with unique projective solution [1 : mth root of lc : 0]
        return q + 1

    if (q-1) % m != 0:
        raise NotImplementedError("Only implemented when m divides q-1 or is coprime to q-1 (automatic if m is prime)")

    e = ZZ((q-1)/m) # a is an mth power if and only if a^e == 1
    
    # We start with the points at infinity
    if d != m:
        count = 1 # [0:1:0] or [1:0:0] as above
    elif F(f[m])^e == 1:
        count = m # [1:mth root of lc:0]
    else:
        count = 0 # [1:mth root of lc:0]

    # Then the affine points
    for x in F:
        fx_to_e = F(f(x))^e
        if fx_to_e == 0:
            count += 1
        elif fx_to_e == 1:
            count += m

    return count


def count_points_hyp(f, q):
    """
    Gives the number of points of the projective
    hyperelliptic curve y^2 = f(x).
    """
    d = f.degree()
    F = GF(q)
    
    assert q % 2 == 1

    # We start with the points at infinity
    if d % 2 == 0:
        # even degree
        lc = F(f[m])
        if lc.is_square():
            count = 2
        else:
            count = 0
    else:
        count = 1

    # Then the affine points
    for x in F:
        fx = F(f(x))
        if fx == 0:
            count += 1
        elif fx.is_square():
            count += 2

    return count


def count_points(m, f, q):
    """
    Gives the number of points of the curve y^m = f(x).
    If m = 2, then we interpret this as a projective hyperelliptic curve.
    Otherwise we interpret it as a plane projective curve of degree max(m, deg(f)+1).
    Note that we do not resolve singularities, so if the plane curve
    y^m = f(x) is singular, then the output can be wrong.
    """
    if m == 2:
        return count_points_hyp(f, q)
    return count_points_plane(m, f, q)


R.<t> = QQ[[]]

def zeta(N):
    """
    Given N = [N1,N2,N3,...,Nk], computes as many terms as possible
    of the zeta function of the curve C. Here N_i = #C(F_{q^i}).
    Note: N[i] = N_{i+1}.
    """
    G = sum([N[l]/(l+1) * t^(l+1) for l in range(len(N))]) + O(t^(len(N)+1)) # k = l+1
    return exp(G)


def frob_char_poly_from_N(N, q, g):
    """
    Given N, q and the genus g of C, where N, q, and C
    are as in the documentation of the function zeta(N), 
    returns the characteristic polynomial of Frobenius of the Jacobian of C.
    """
    Z = zeta(N)
    f = Z * (1-t) * (1-q*t)
    if f.prec() <= g:
        raise ValueError("not enough terms")
    for i in range(2*g+1, f.prec()):
        if f[i] != 0:
            raise ValueError("invalid sequence for this genus")
    x = polygen(ZZ)
    p = sum([f[i]*q^(g-i)*x^i for i in range(g+1)]) + sum([f[i]*x^(2*g-i) for i in range(g)])
    for i in range(f.prec()):
        if p[i]*q^(i-g) != f[i]:
            raise ValueError("invalid sequence for this genus")
    return p


def frob_char_poly(m, f, q, extra_terms=1):
    """
    Given m, f, and q as in the documentation of count_points,
    returns the characterstic polynomial of Frobenius of the Jacobian
    of the curve C : y^m = f(x) over F_q.
    
    If extra_terms is positive, then compute some redundant terms.
    
    Assumes that the given affine model of a hyperelliptic curve,
    or projective model otherwise, is smooth.
    """
    if m == 2:
        g = ceil((f.degree()-2)/2)
    else:
        d = max(m, f.degree())
        g = (d-1)*(d-2)/2
    N = [count_points(m, f, q^i) for i in range(1, 7+extra_terms)]
    return frob_char_poly_from_N(N, q, g)

    
def newton_polygon(m, f, p, extra_terms=0):
    """
    Given m, f, p=q, and extra_terms as in the documentation of frob_char_poly,
    computes the Newton polygon. Assumes that p is prime.
    
    The output is:
        - pts, the y-coordinates of the points of the Newton polygon
        - slopes, the slopes between these points
        - a human-readable version of the slopes:
            - 'ss' if the curve is supersingular
            - 'ord' if the curve is ordinary
            - a copy of slopes otherwise
        
    """
    poly = frob_char_poly(m, f, p, extra_terms=extra_terms)
    g = poly.degree() / 2
    pts = [(i, poly[2*g-i].valuation(p)) for i in range(2*g+1) if poly[i] != 0]
    pts2 = [oo for i in range(2*g+1)]
    n = len(pts)
    for j in range(n):
        (x2,y2) = pts[j]
        for i in range(j):
            (x1,y1) = pts[i]
            for x in range(x1, x2+1):
                y = y1 + (x-x1)*(y2-y1)/(x2-x1)
                if y < pts2[x]:
                    pts2[x] = y
    slopes = [pts2[i+1]-pts2[i] for i in range(2*g)]
    return pts2, slopes, 'ss' if slopes == [1/2 for i in range(2*g)] else ('ord' if slopes == [0 for i in range(g)] + [1 for i in range(g)] else slopes)
