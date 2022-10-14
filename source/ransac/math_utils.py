import math
import torch


def multi_cubic(a0, b0, c0, d0, all_roots=True):
    ''' Analytical closed-form solver for multiple cubic equations
    (3rd order polynomial), based on `numpy` functions.
    Parameters
    ----------
    a0, b0, c0, d0: array_like
        Input data are coefficients of the Cubic polynomial::
            a0*x^3 + b0*x^2 + c0*x + d0 = 0
    all_roots: bool, optional
        If set to `True` (default) all three roots are computed and returned.
        If set to `False` only one (real) root is computed and returned.
    Returns
    -------
    roots: ndarray
        Output data is an array of three roots of given polynomials of size
        (3, M) if `all_roots=True`, and an array of one root of size (M,)
        if `all_roots=False`.
    '''

    ''' Reduce the cubic equation to to form:
        x^3 + a*x^2 + bx + c = 0'''
    a, b, c = b0 / a0, c0 / a0, d0 / a0

    device = a0.device

    # Some repeating constants and variables
    third = 1./3.
    a13 = a*third
    a2 = a13*a13
    sqr3 = math.sqrt(3)

    # Additional intermediate variables
    f = third*b - a2
    g = a13 * (2*a2 - b) + c
    h = 0.25*g*g + f*f*f

    # Masks for different combinations of roots
    m1 = (f == 0) & (g == 0) & (h == 0)     # roots are real and equal
    m2 = (~m1) & (h <= 0)                   # roots are real and distinct
    m3 = (~m1) & (~m2)                      # one real root and two complex

    def cubic_root(x):
        ''' Compute cubic root of a number while maintaining its sign
        '''
        root = torch.zeros_like(x)
        positive = (x >= 0)
        negative = ~positive
        root[positive] = x[positive]**third
        root[negative] = -(-x[negative])**third
        return root

    def roots_all_real_equal(c):
        ''' Compute cubic roots if all roots are real and equal
        '''
        r1 = -cubic_root(c).type(torch.cfloat)
        if all_roots:
            return torch.stack((r1, r1, r1), dim=0)
        else:
            return r1

    def roots_all_real_distinct(a13, f, g, h):
        ''' Compute cubic roots if all roots are real and distinct
        '''
        j = torch.sqrt(-f)
        k = torch.arccos(-0.5*g / (j*j*j))
        m = torch.cos(third*k)
        r1 = 2*j*m - a13
        if all_roots:
            n = sqr3 * torch.sin(third*k)
            r2 = -j * (m + n) - a13
            r3 = -j * (m - n) - a13
            return torch.stack((r1, r2, r3), dim=0).type(torch.cfloat)
        else:
            return r1

    def roots_one_real(a13, g, h):
        ''' Compute cubic roots if one root is real and other two are complex
        '''
        sqrt_h = torch.sqrt(h)
        S = cubic_root(-0.5*g + sqrt_h)
        U = cubic_root(-0.5*g - sqrt_h)
        S_plus_U = S + U
        r1 = S_plus_U - a13
        if all_roots:
            S_minus_U = S - U
            r2 = -0.5*S_plus_U - a13 + S_minus_U*sqr3*0.5j
            r3 = -0.5*S_plus_U - a13 - S_minus_U*sqr3*0.5j
            return torch.stack((r1, r2, r3), dim=0).type(torch.cfloat)
        else:
            return r1

    # Compute roots
    if all_roots:
        roots = torch.zeros((3, len(a)), device=device, dtype=torch.cfloat)
        roots[:, m1] = roots_all_real_equal(c[m1])
        roots[:, m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
        roots[:, m3] = roots_one_real(a13[m3], g[m3], h[m3])
    else:
        roots = torch.zeros(len(a), device=device, dtype=torch.cfloat)  
        roots[m1] = roots_all_real_equal(c[m1])
        roots[m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
        roots[m3] = roots_one_real(a13[m3], g[m3], h[m3])

    return roots


class StrumPolynomialSolver(object):
    def __init__(self, n):
        self.n = n

    def build_sturm_seq(self, fvec):
        #fvec = fvec + 2 * self.n + 1
        f = torch.zeros(3 * self.n, dtype=torch.float64, device=fvec.device)
        f[:2 * self.n + 1] = fvec
        f[2 * self.n + 1:] = torch.tensor([-9.2559631349317831e+61]*(self.n-1), dtype=torch.float64)
        f1 = 0
        f2 = self.n + 1
        f3 = 2 * self.n + 1
        svec = torch.zeros(3 * self.n, dtype=torch.float64, device=fvec.device)

        for i in range(self.n - 1):
            q1 = f[f1 + self.n - i] * f[f2 + self.n - 1 - i]
            q0 = f[f1 + self.n - 1 - i] * f[f2 + self.n - 1 - i] - f[f1 + self.n - i] * f[f2 + self.n - 2 - i]

            f[f3] = f[f1] - q0 * f[f2]
            for j in range(1, self.n - 1 - i):
                f[f3 + j] = f[f1 + j] - q1 * f[f2 + j - 1] - q0 * f[f2 + j]

            c = -abs(f[f3 + self.n - 2 - i])
            for j in range(0, self.n - 1 - i):
                f[f3 + j] = f[f3 + j] * (1 / c)

            # juggle pointers(f1, f2, f3) -> (f2, f3, f1)
            tmp = f1
            f1, f2, f3 = f2, f3, tmp

            # svec = torch.stack(q0, q1, c) # columns
            svec[3 * i] = q0
            svec[3 * i + 1] = q1
            svec[3 * i + 2] = c

        svec[3 * self.n - 3] = f[f1]
        svec[3 * self.n - 2] = f[f1 + 1]
        svec[3 * self.n - 1] = f[f2]

        return svec

    def get_bounds(self, fvec):
        max_ = 0
        for i in range(self.n):
            max_ = max([max_, abs(fvec[i])])
        return 1 + max_

    def flag_negative(self, f, n):
        if n <= 0:
            return f[0] < 0
        else:
            return (int(f[n] < 0) << n) | self.flag_negative(f, n-1)  # '<<' will cause lshift_cuda' not implemented for bool

    def change_sign(self, svec, x):
        f = torch.tensor([-9.2559631349317831e+61]*(self.n + 1), dtype=torch.float64, device=svec.device)
        f[self.n] = svec[3 * self.n - 1]
        f[self.n - 1] = svec[3 * self.n - 3] + x * svec[3 * self.n - 2]
        for i in range(self.n - 2, -1, -1):
            f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2]

        # negative flag
        S = self.flag_negative(f, self.n)

        return self.NumberOf1((S ^ (S >> 1)) & ~(0xFFFFFFFF << self.n))

    def NumberOf1(self, n):
        return bin(n & 0xffffffff).count('1')

    def polyval(self, f, x, n):
        fx = x + f[n - 1]

        for i in range(n - 2, -1, -1):
            fx = x * fx + f[i]

        return fx

    def ridders_method_newton(self, fvec, a, b, tol, tol_newton=1e-3/2, n_roots=None):
        """
            Applies Ridder's bracketing method until we get close to root, followed by newton iterations

        """

        fa = self.polyval(fvec, a, self.n)
        fb = self.polyval(fvec, b, self.n)

        if not((fa * fb) < 0):
             return 0, 0

        for i in range(30):
            if abs(a - b) < tol_newton:
                break

            c = (a + b) * 1 / 2
            fc = self.polyval(fvec, c, self.n)

            s = torch.sqrt(fc ** 2 - fa * fb)
            if not s:
                break

            d = c + (a - c) * fc / s if (fa < fb) else c + (c - a) * fc / s

            fd = self.polyval(fvec, d, self.n)

            if ( (fc < 0) if (fd >= 0) else (fc > 0)):
                a = c
                fa = fc
                b = d
                fb = fd
            elif ((fa < 0) if (fd >= 0) else (fa > 0)):
                b = d
                fb = fd
            else:
                a = d
                fa = fd

        # We switch to Newton's method once we are close to the root
        x = (a + b) * 0.5
        fpvec = fvec[self.n + 1:]
        for i in range(0, 10):
            fx = self.polyval(fvec, x, self.n)
            if abs(fx) < tol:
                break
            fpx = self.n * self.polyval(fpvec, x, self.n - 1)
            dx = fx / fpx
            x = x - dx
            if abs(dx) < tol:
                break
        #roots[n_roots] = x.clone()
        n_roots += 1

        return n_roots, x

    def isolate_roots(self, fvec, svec, a, b, sa, sb, tol, depth, n_roots=None, roots=None):

        if depth > 30:
            return 0, roots

        if (sa - sb) > 1:
            c = 1 / 2 * (a + b)
            sc = self.change_sign(svec, c)
            n_roots, roots = self.isolate_roots(fvec, svec, a, c, sa, sc, tol, depth + 1, n_roots=n_roots, roots=roots)
            n_roots, roots = self.isolate_roots(fvec, svec, c, b, sc, sb, tol, depth + 1, n_roots=n_roots, roots=roots)

        elif (sa - sb) == 1:
            n_roots, x = self.ridders_method_newton(fvec, a, b, tol, n_roots=n_roots)
            roots[n_roots - 1] = x

        return n_roots, roots

    def bisect_sturm(self, coeffs, tol=1e-10):
        if (coeffs[self.n-1] == 0.0):
            return 0, None

        # fvec is the polynomial and its first derivative.
        fvec = torch.zeros(2 * self.n + 1, dtype=torch.float64, device=coeffs.device)
        fvec[:self.n + 1], fvec[self.n + 1:] = coeffs, torch.tensor([-9.2559631349317831e+61] * (self.n),
                                                                          dtype=torch.float64)
        fvec[:self.n + 1] *= 1 / fvec[self.n]
        fvec[self.n] = 1
        # fvec = torch.zeros(coeffs.shape[0], 2 * self.n + 1, dtype=torch.float64, device=coeffs.device)
        # fvec[:, :self.n+1], fvec[:, self.n+1:] = coeffs, torch.tensor([-9.2559631349317831e+61]*(self.n), dtype=torch.float64)
        # fvec[:, self.n+1] *= 1 / fvec[:, self.n]
        # fvec[:, self.n] = 1

        # Compute the derivative with normalized coefficients
        for i in range(self.n - 1):
            fvec[self.n + 1 + i] = fvec[i + 1] * ((i + 1) / self.n)
        fvec[2 * self.n] = 1

        # Compute sturm sequences
        svec = self.build_sturm_seq(fvec)

        # All real roots are in the interval [-r0, r0]
        r0 = self.get_bounds(fvec)
        sa = self.change_sign(svec, -r0)
        sb = self.change_sign(svec, r0)

        n_roots = sa - sb
        if n_roots <= 0:
            return 0, None
        roots = torch.zeros(n_roots, device=fvec.device)
        n_roots = 0
        n_roots, roots = self.isolate_roots(fvec, svec, -r0, r0, sa, sb, tol, 0, n_roots=n_roots, roots=roots)
        # print(n_roots)
        return n_roots, roots
#
# s = StrumPolynomialSolver(10)
#
# coeffs = torch.tensor([-0.046221584891460658, -1.2302772490478291, -11.092469777927988, -36.8025470317145752, -7.3000058107138752,
#                        101.03357698261544, -90.724155357707190, 108.03629828827744, -66.526889338835147,
#                        3.5187562174661053, 0.34904922215607215], dtype=torch.float64, requires_grad=True)
# ret, roots = s.bisect_sturm(coeffs)
# target = torch.rand(roots.shape, device=roots.device)
# loss = torch.norm(roots-target)#min_matches
# loss.backward()
# coeffs.grad
# # print(ret, roots)

class StrumPolynomialSolverBatch(object):
    def __init__(self, n, batch_size):
        self.n = n
        self.batch_size = batch_size
    def build_sturm_seq(self, fvec):
        #fvec = fvec + 2 * self.n + 1

        f = torch.zeros(self.batch_size, 3 * self.n, dtype=torch.float64, device=fvec.device)
        f[:, :2 * self.n + 1] = fvec
        f[:, 2 * self.n + 1:] = torch.tensor([-9.2559631349317831e+61]*(self.batch_size*(self.n-1)), dtype=torch.float64).view(self.batch_size, -1)

        f1 = 0
        f2 = self.n + 1
        f3 = 2 * self.n + 1
        svec = torch.zeros(self.batch_size, 3 * self.n, dtype=torch.float64, device=fvec.device)

        for i in range(self.n - 1):
            q1 = f[:, f1 + self.n - i] * f[:, f2 + self.n - 1 - i]
            q0 = f[:, f1 + self.n - 1 - i] * f[:, f2 + self.n - 1 - i] - f[:, f1 + self.n - i] * f[:, f2 + self.n - 2 - i]

            f[:, f3] = f[:, f1] - q0 * f[:, f2]
            for j in range(1, self.n - 1 - i):
                f[:, f3 + j] = f[:, f1 + j] - q1 * f[:, f2 + j - 1] - q0 * f[:, f2 + j]

            c = -abs(f[:, f3 + self.n - 2 - i])
            for j in range(0, self.n - 1 - i):
                f[:, f3 + j] = f[:, f3 + j] * (1 / c)

            # juggle pointers(f1, f2, f3) -> (f2, f3, f1)
            tmp = f1
            f1, f2, f3 = f2, f3, tmp

            # svec = torch.stack(q0, q1, c) # columns
            svec[:, 3 * i] = q0
            svec[:, 3 * i + 1] = q1
            svec[:, 3 * i + 2] = c

        svec[:, 3 * self.n - 3] = f[:, f1]
        svec[:, 3 * self.n - 2] = f[:, f1 + 1]
        svec[:, 3 * self.n - 1] = f[:, f2]

        return svec

    def get_bounds(self, fvec):
        max_, _ = torch.max(abs(fvec[:, :10]), dim=-1)
        # max_ = 0
        # for i in range(self.n):
        #     max_ = max([max_, abs(fvec[i])])
        return 1 + max_

    def flag_negative(self, f, n):
        if n <= 0:
            return f[0] < 0
        else:
            return (int(f[n] < 0) << n) | self.flag_negative(f, n-1)

    def change_sign(self, svec, x):
        f = torch.tensor([-9.2559631349317831e+61]*(self.n + 1), dtype=torch.float64, device=svec.device)
        f[self.n] = svec[3 * self.n - 1]
        f[self.n - 1] = svec[3 * self.n - 3] + x * svec[3 * self.n - 2]
        for i in range(self.n - 2, -1, -1):
            f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2]

        S = self.flag_negative(f, self.n)

        return self.NumberOf1((S ^ (S >> 1)) & ~(0xFFFFFFFF << self.n))

    def change_sign_batch(self, svec, x):
        f = torch.tensor([-9.2559631349317831e+61]*(self.batch_size*(self.n + 1)), dtype=torch.float64, device=svec.device).view(self.batch_size, -1)
        f[:, self.n] = svec[:, 3 * self.n - 1]
        f[:, self.n - 1] = svec[:, 3 * self.n - 3] + x * svec[:, 3 * self.n - 2]
        for i in range(self.n - 2, -1, -1):
            f[:, i] = (svec[:, 3 * i] + x * svec[:, 3 * i + 1]) * f[:, i + 1] + svec[:, 3 * i + 2] * f[:, i + 2]

        ret = []
        # negative flag
        for i in range(f.shape[0]):
            S = self.flag_negative(f[i], self.n)
            ret.append(torch.tensor(self.NumberOf1((S ^ (S >> 1)) & ~(0xFFFFFFFF << self.n)), device=f.device))

        return torch.stack(ret)
    def NumberOf1(self, n):
        return bin(n & 0xffffffff).count('1')

    def polyval(self, f, x, n):
        # n = 3
        # x + f[2]
        # x2 + f[2] * x + f[1]
        # x3 + f[2] * x2 + f[1] * x + f[0]
        fx = x + f[n - 1]

        for i in range(n - 2, -1, -1):
            fx = x * fx + f[i]

        return fx

    def ridders_method_newton(self, fvec, a, b, tol, tol_newton=1e-3/2, n_roots=None):
        """
            Applies Ridder's bracketing method until we get close to root, followed by newton iterations

        """

        fa = self.polyval(fvec, a, self.n)
        fb = self.polyval(fvec, b, self.n)

        if not((fa * fb) < 0):
             return 0, torch.zeros(1, device=fvec.device)

        for i in range(30):
            if abs(a - b) < tol_newton:
                break

            c = (a + b) * 1 / 2
            fc = self.polyval(fvec, c, self.n)

            s = torch.sqrt(fc ** 2 - fa * fb)
            if not s:
                break

            d = c + (a - c) * fc / s if (fa < fb) else c + (c - a) * fc / s

            fd = self.polyval(fvec, d, self.n)

            if ( (fc < 0) if (fd >= 0) else (fc > 0)):
                a = c
                fa = fc
                b = d
                fb = fd
            elif ((fa < 0) if (fd >= 0) else (fa > 0)):
                b = d
                fb = fd
            else:
                a = d
                fa = fd

        # We switch to Newton's method once we are close to the root
        x = (a + b) * 0.5
        fpvec = fvec[self.n + 1:]
        for i in range(0, 10):
            fx = self.polyval(fvec, x, self.n)
            if abs(fx) < tol:
                break
            fpx = self.n * self.polyval(fpvec, x, self.n - 1)
            dx = fx / fpx
            x = x - dx
            if abs(dx) < tol:
                break
        #roots[n_roots] = x.clone()
        n_roots += 1

        return n_roots, x

    def isolate_roots(self, fvec, svec, a, b, sa, sb, tol, depth, n_roots=None, roots=None):

        if depth > 30:
            return 0, roots

        if (sa - sb) > 1:
            c = 1 / 2 * (a + b)
            sc = self.change_sign(svec, c)
            n_roots, roots = self.isolate_roots(fvec, svec, a, c, sa, sc, tol, depth + 1, n_roots=n_roots, roots=roots)
            n_roots, roots = self.isolate_roots(fvec, svec, c, b, sc, sb, tol, depth + 1, n_roots=n_roots, roots=roots)

        elif (sa - sb) == 1:
            try:
                n_roots, x = self.ridders_method_newton(fvec, a, b, tol, n_roots=n_roots)
            except ValueError:
                print("")


            roots[n_roots - 1] = x

        return n_roots, roots

    def bisect_sturm(self, coeffs, custom_sols=0, tol=1e-10):
        # if (coeffs[self.n-1] == 0.0):
        #     return 0

        # fvec is the polynomial and its first derivative.
        fvec = torch.zeros(self.batch_size, 2 * self.n + 1, dtype=torch.float64, device=coeffs.device)
        fvec[:, :self.n+1], fvec[:, self.n+1:] = coeffs, torch.tensor([-9.2559631349317831e+61]*(self.batch_size*self.n),
                                                                      dtype=torch.float64).view(self.batch_size, -1)
        fvec[:, :self.n+1] = (torch.bmm((fvec[:, :self.n+1].clone()).unsqueeze(-1), (1 / fvec[:, self.n].clone()).unsqueeze(-1).unsqueeze(-1))).squeeze(-1)
        fvec[:, self.n] = torch.ones(self.batch_size)

        # Compute the derivative with normalized coefficients
        for i in range(self.n - 1):
            fvec[:, self.n + 1 + i] = fvec[:, i + 1] * ((i + 1) / self.n)

        fvec[:, 2 * self.n] = torch.ones(self.batch_size)

        # Compute sturm sequences
        svec = self.build_sturm_seq(fvec)

        # All real roots are in the interval [-r0, r0]
        r0 = self.get_bounds(fvec)
        sa = self.change_sign_batch(svec, -r0)
        sb = self.change_sign_batch(svec, r0)

        #root_dict = {}#dict.fromkeys(zip(list(range(self.batch_size))))
        roots = []
        if custom_sols == 0:
            n_roots = sa - sb
        else:
            n_roots = torch.ones(self.batch_size, device=fvec.device) * custom_sols

        for i in range(self.batch_size):
            if n_roots[i] <= 0:
                root = torch.zeros(1, device=fvec.device)
            else:
                root = torch.zeros(int(n_roots[i]), device=fvec.device)

                n_root = 0
                n_root, root = self.isolate_roots(fvec[i], svec[i], -r0[i], r0[i], sa[i], sb[i], tol, 0, n_roots=n_root, roots=root)
            roots.append(root)

        return n_roots, roots
"""
s = StrumPolynomialSolverBatch(10, 2)

coeffs = torch.tensor([[-0.046221584891460658, -1.2302772490478291, -11.092469777927988, -36.8025470317145752, -7.3000058107138752,
                       101.03357698261544, -90.724155357707190, 108.03629828827744, -66.526889338835147,
                       3.5187562174661053, 0.34904922215607215],
                      [-0.046221584891460658, -1.2302772490478291, -11.092469777927988, -36.8025470317145752,
                       -7.3000058107138752,
                       101.03357698261544, -90.724155357707190, 108.03629828827744, -66.526889338835147,
                       3.5187562174661053, 0.34904922215607215]], dtype=torch.float64, device='cuda:0', requires_grad=True)
root_dicts = s.bisect_sturm(coeffs)
with torch.autograd.set_detect_anomaly(True):
    for i in range(len(root_dicts.keys())):
        target = torch.rand(root_dicts[i][-1].shape, device=root_dicts[i][-1].device)
        loss = torch.norm(root_dicts[i][-1]-target)#min_matches
        loss.backward()
        coeffs.grad
# # print(ret, roots)
"""


def ned_torch(x1: torch.Tensor, x2: torch.Tensor, dim=1, eps=1e-8) -> torch.Tensor:
    """
    Normalized euclidean distance in pytorch.
    https://github.com/brando90/Normalized-Euclidean-Distance-and-Similarity-NED-NES-/blob/main/main_ned_nes.py
    Cases:
        1. For comparison of two vecs directly make sure vecs are of size [B] e.g. when using nes as a loss function.
            in this case each number is not considered a representation but a number and B is the entire vector to
            compare x1 and x2.
        2. For comparison of two batch of representation of size 1D (e.g. scores) make sure it's of shape [B, 1].
            In this case each number *is* the representation of the example. Thus a collection of reps
            [B, 1] is mapped to a rep of size [B]. Note usually D does decrease since reps are not of size 1
            (see case 3)
        3. For the rest specify the dimension. Common use case [B, D] -> [B, 1] for comparing two set of
            activations of size D. In the case when D=1 then we have [B, 1] -> [B, 1]. If you meant x1, x2 [D, 1] to be
            two vectors of size D to be compare feed them with shape [D].
            This one is also good for computing the NED for two batches of values. e.g. if you have a tensor of size
            [B, k] and the row is a batch and each entry is the y value for that batch. If a batch is a task then
            it is computing the NED for that task, which is good because that batch has it's own unique scale that
            we are trying to normalize and by doing it per task you are normalizing it as you'd expect (i.e. per task).
    Note: you cannot use this to compare two single numbers NED(x,y) is undefined because a single number does not have
    a variance. Variance[x] = undefined.
    https://discuss.pytorch.org/t/how-does-one-compute-the-normalized-euclidean-distance-similarity-in-a-numerically-stable-way-in-a-vectorized-way-in-pytorch/110829
    https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance/498753?noredirect=1#comment937825_498753
    https://github.com/brando90/Normalized-Euclidean-Distance-and-Similarity
    """
    # to compute ned for two individual vectors e.g to compute a loss (NOT BATCHES/COLLECTIONS of vectorsc)
    if len(x1.size()) == 1:
        # [K] -> [1]
        ned_2 = 0.5 * ((x1 - x2).var() / (x1.var() + x2.var() + eps))
    # if the input is a (row) vector e.g. when comparing two batches of acts of D=1 like with scores right before sf
    elif x1.size() == torch.Size([x1.size(0), 1]):  # note this special case is needed since var over dim=1 is nan (1 value has no variance).
        # [B, 1] -> [B]
        ned_2 = 0.5 * ((x1 - x2)**2 / (x1**2 + x2**2 + eps)).squeeze()  # Squeeze important to be consistent with .var, otherwise tensors of different sizes come out without the user expecting it
    # common case is if input is a batch
    else:
        # e.g. [B, D] -> [B]
        ned_2 = 0.5 * ((x1 - x2).var(dim=dim) / (x1.var(dim=dim) + x2.var(dim=dim) + eps))
    return ned_2 ** 0.5
