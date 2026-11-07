# 2. Equivalent formulations of RH (with precise statements)

Each item below is a theorem of the form "RH $\iff$ (statement)". For each we say what is implemented in this repository. Notation: $\rho$ runs over the nontrivial zeros, $\gamma=\Im\rho$, $\gamma_E$ is Euler's constant, $\sigma(n)=\sum_{d\mid n}d$, $\varphi$ is Euler's function, $\mu$ Möbius, $\Lambda$ von Mangoldt, $H_n=\sum_{k\le n}1/k$.

## 2.1 Prime-counting and summatory functions

1. **von Koch (1901).** RH $\iff \psi(x)=x+O(\sqrt x\log^2x) \iff \pi(x)=\mathrm{li}(x)+O(\sqrt x\log x)$. Explicitly (Schoenfeld 1976) RH $\Rightarrow |\pi(x)-\mathrm{li}(x)|<\frac{\sqrt x\log x}{8\pi}$ for $x\ge2657$, and $|\psi(x)-x|<\frac{\sqrt x\log^2x}{8\pi}$ for $x\ge73.2$.
2. **Littlewood.** RH $\iff M(x)=\sum_{n\le x}\mu(n)=O_\varepsilon(x^{1/2+\varepsilon})$ for every $\varepsilon>0$. (Equivalently, by Redheffer, $\det R_n = M(n)$ where $R_n$ is the $n\times n$ 0/1 Redheffer matrix.)
3. **Franel–Landau (1924).** Let $r_1<\dots<r_{\Phi(n)}$ be the Farey fractions of order $n$. RH $\iff \sum_\nu |r_\nu-\nu/\Phi(n)| = O(n^{1/2+\varepsilon})$.
4. **Riesz (1916).** RH $\iff \sum_{k\ge1}\frac{(-1)^{k+1}x^k}{(k-1)!\,\zeta(2k)}=O(x^{1/4+\varepsilon})$. **Hardy–Littlewood (1918):** RH $\iff \sum_{k\ge1}\frac{(-x)^k}{k!\,\zeta(2k+1)}=O(x^{-1/4+\varepsilon})$ as $x\to\infty$ (this series *decays*; the exponent is $-1/4$).
5. **Báez-Duarte's sequential Riesz criterion (2005).** With $c_k=\sum_{j=0}^k(-1)^j\binom kj\frac1{\zeta(2j+2)}$: RH $\iff c_k=O(k^{-3/4+\varepsilon})$. *Implemented: `sequential_criteria.py` (exact values to $k=10^6$ via the Möbius form $c_k=\sum_n\mu(n)n^{-2}(1-n^{-2})^k$, and the explicit formula over zeros).*

## 2.2 Elementary arithmetic inequalities

6. **Robin (1984).** RH $\iff \sigma(n)<e^{\gamma_E}\,n\log\log n$ for all $n>5040$. Unconditionally $\sigma(n)<e^{\gamma_E}n\log\log n+\frac{0.6483\,n}{\log\log n}$ for $n\ge3$. The 27 exceptions $\le5040$ are known; the inequality is verified for $n\le 10^{10^{10}}$ via colossally abundant numbers (Briggs 2006) and in many infinite families (Choie–Lichiardopol–Moree–Solé 2007, …). *Implemented: `arithmetic.robin_check`, `arithmetic.ca_numbers`.*
7. **Lagarias (2002).** RH $\iff \sigma(n)\le H_n+e^{H_n}\log H_n$ for all $n\ge1$, with equality only for $n=1$. *Implemented: `arithmetic.lagarias_check`.*
8. **Nicolas (1983).** With $N_k=p_1\cdots p_k$: RH $\iff \frac{N_k}{\varphi(N_k)}>e^{\gamma_E}\log\log N_k$ for all $k\ge1$. If RH fails, the inequality holds for infinitely many $k$ and fails for infinitely many $k$. *Implemented: `arithmetic.nicolas_check`.*

## 2.3 Criteria on the zeros / on $\xi$

9. **Li (1997).** $\lambda_n:=\frac1{(n-1)!}\frac{d^n}{ds^n}\bigl[s^{n-1}\log\xi(s)\bigr]_{s=1}=\sum_\rho\bigl[1-(1-\tfrac1\rho)^n\bigr]$. RH $\iff\lambda_n>0$ for all $n\ge1$. Bombieri–Lagarias (1999) gave the arithmetic formula for $\lambda_n$ and showed the criterion is a special case of Weil's positivity (item 12). Under RH, $\lambda_n=\frac n2\log n+\frac n2(\gamma_E-1-\log2\pi)+O(\sqrt n\log n)$ (a theorem: Lagarias 2007, Thm 1.1; Voros 2006 for sharper forms). *Implemented: `li.py` (zeros + tail, and an independent high-precision power-series method).*
10. **Speiser (1934).** RH $\iff\zeta'(s)\ne0$ for $0<\Re s<\tfrac12$. Levinson–Montgomery (1974): $\zeta$ and $\zeta'$ have, up to $O(\log T)$, the same number of zeros in $0<\sigma<1/2$, $0<t<T$. *Implemented: `speiser.py`.*
11. **de Bruijn–Newman.** RH $\iff\Lambda\le0$; with Rodgers–Tao, RH $\iff\Lambda=0$. *Implemented (dynamics, Lehmer pairs): `debruijn_newman.py`.*
12. **Weil positivity (Weil 1952; Bombieri 2000).** For $g\in C_c^\infty(\mathbb R)$ put $\tilde g(x)=\overline{g(-x)}$ and $\hat g(r)=\int g(u)e^{iru}du$. Define the Weil distribution
$$W(g)=\sum_\rho \hat g\!\left(\tfrac{\rho-1/2}{i}\right)=\hat g(\tfrac i2)+\hat g(-\tfrac i2)-\sum_{n}\frac{\Lambda(n)}{\sqrt n}\bigl(g(\log n)+g(-\log n)\bigr)+\frac1{2\pi}\int_{\mathbb R}\hat g(r)\Bigl(\Re\psi\bigl(\tfrac14+\tfrac{ir}2\bigr)-\log\pi\Bigr)dr .$$
Then RH $\iff W(g*\tilde g)\ge0$ for all $g$. (The zero side is $\sum_\rho|\hat g(\ldots)|^2\ge0$ trivially *if* the zeros are on the line; the content is positivity of the arithmetic/archimedean right-hand side.) Li's criterion is the case of the test functions $\hat g_n(r)=1-(1-1/\rho)^n$ in a suitable sense (Bombieri–Lagarias). *Implemented (numerical evaluation of both sides, positivity of the arithmetic side for Gaussian families): `explicit_formula.py`.*
13. **Nyman (1950)–Beurling (1955)–Báez-Duarte (2003).** With $\rho_k(x)=\{1/(kx)\}$ in $L^2(0,\infty)$ and
$d_N^2=\inf_{c\in\mathbb R^N}\|\chi_{(0,1]}-\sum_{k\le N}c_k\rho_k\|^2=\inf_{A_N}\frac1{2\pi}\int_{\mathbb R}\frac{|1-\zeta(\tfrac12+it)A_N(\tfrac12+it)|^2}{\tfrac14+t^2}dt$ ($A_N$ Dirichlet polynomial of length $N$): RH $\iff d_N\to0$. Burnol (2002): $\liminf d_N^2\log N\ge\sum_{\rho\ \text{on the line}}\frac{m_\rho^2}{|\rho|^2}$ ($=2+\gamma_E-\log4\pi=0.0461914\ldots$ if all zeros are simple and on the line). Conjecture (BDBLS 2000): $d_N^2\sim\frac{2+\gamma_E-\log4\pi}{\log N}$. *Implemented with the verified Vasyunin closed form for the Gram matrix and both float64 and rigorous ball-arithmetic solves: `nyman_beurling.py`.*
14. **Balazard–Saias–Yor (1999).** RH $\iff\displaystyle\int_{\Re s=1/2}\frac{\log|\zeta(s)|}{|s|^2}\,|ds|=0$. (In general the integral equals $2\pi\sum_{\Re\rho>1/2}\log\left|\frac{\rho}{1-\rho}\right|\ge0$.) *Implemented: `integral_criteria.bsy_integral`.*
15. **Volchkov (1995).** RH $\iff\displaystyle\int_0^\infty\frac{1-12t^2}{(1+4t^2)^3}\int_{1/2}^\infty\log|\zeta(\sigma+it)|\,d\sigma\,dt=\pi\frac{3-\gamma_E}{32}$. *Implemented: `integral_criteria.volchkov_integral`.*
16. **Lagarias' $\mathrm{Re}(\xi'/\xi)$ criterion (1999).** RH $\iff \Re\frac{\xi'}{\xi}(s)>0$ for $\Re s>\tfrac12$ (equivalently $\frac{\xi'}{\xi}$ maps the right half-plane into itself; from the Hadamard product, $\Re\frac{\xi'}{\xi}(s)=\sum_\rho\Re\frac1{s-\rho}$).
17. **Hilbert space of Dirichlet series / Beurling's version.** RH $\iff$ the functions $\{\rho_\theta:0<\theta\le1\}$ span $L^2(0,1)$; in Mellin terms (item 13), $\iff$ the constant $1$ lies in the closure of $\{\zeta(s)A(s): A\ \text{Dirichlet polynomial}\}$ in $L^2(\Re s=\tfrac12,\ dt/(\tfrac14+t^2))$. (Note that $1/\zeta$ itself is *not* in that space: it has non-integrable singularities at the zeros.)

## 2.4 Spectral reformulations (conjectural equivalents, not theorems)

* **Hilbert–Pólya.** Write the nontrivial zeros as $\rho=\tfrac12+i\tau_\rho$ with $\tau_\rho\in\mathbb C$ (so $\tau_\rho$ is real exactly when $\rho$ is on the line). If there is a self-adjoint operator $H$ with spectrum $\{\tau_\rho\}$ then RH holds (eigenvalues of a self-adjoint operator are real). The converse direction is empty: given RH one can always define a diagonal operator with the $\gamma_n$; the content is to find an $H$ with an *independent* definition (geometric, arithmetic, or dynamical) whose trace formula is the explicit formula.
* **Connes (1999).** RH for all $L(s,\chi)$ $\iff$ a trace formula on the adele class space $\mathbb A_\mathbb Q/\mathbb Q^*$ holds / the Weil positivity holds; the Weil distribution is realised as a trace.
* **Deninger.** A conjectural cohomology with $\xi(s)=\det_\infty(\frac{1}{2\pi}(s-\Theta)|H^1)$; RH would follow from the self-adjointness of $\Theta$ with respect to a positive form (an analogue of the Hodge index theorem).

## 2.5 Which of these this repository computes

| Criterion | Module | Rigorous? |
|---|---|---|
| zeros on the line, $N(T)$ | `zeta.py`, `verify.py` (independent Riemann–Siegel + Turing), `data.py` (Arb) | Arb zeros: yes; Turing in float64: certified modulo float rounding |
| Li $\lambda_n>0$ | `li.py` | high-precision series for small $n$ (yes); zero-sum with tail for large $n$ (error estimated) |
| Robin / Lagarias / Nicolas / CA numbers | `arithmetic.py` | $\sigma(n)$ exact (integer sieve); the inequalities are decided in float64/long double; the smallest relative margins up to $10^7$ are $8.7\times10^{-4}$ (Robin, $n=720$) and $1.1\times10^{-2}$ (Lagarias, $n=12$), against rounding $\lesssim10^{-15}$, and any decision within $10^{-10}$ is re-decided in Arb; CA numbers exact, their ratios at 60 digits |
| Speiser | `speiser.py` | numerical root finding + argument principle |
| Nyman–Beurling $d_N$ | `nyman_beurling.py` | Arb ball enclosures for $N\le$ several hundred; float64 to $N\sim10^4$ with condition-number error bound |
| Weil explicit formula / positivity | `explicit_formula.py` | numerical (truncation errors estimated) |
| de Bruijn–Newman dynamics | `debruijn_newman.py` | numerical ODE |
| Davenport–Heilbronn (RH fails without Euler product) | `davenport_heilbronn.py` | off-line zeros certified by the Krawczyk test (rigorous $\Re s>\tfrac12$, pairwise distinct); on-line zeros by sign changes of the real Hardy-type function; total count by the argument principle |
| GUE statistics | `statistics.py` | numerical |
| Balazard–Saias–Yor, Volchkov integrals | `integral_criteria.py` | numerical (tanh-sinh, Arb $\zeta$), identities verified to $\sim10^{-9}$ / $10^{-7}$ |
| Báez-Duarte sequential $c_k$ | `sequential_criteria.py` | float64/mpmath exact sums; explicit formula over zeros |
