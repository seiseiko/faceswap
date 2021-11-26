import cv2
import time
import numpy as np
from math import floor, ceil
import scipy
from numpy.fft import fft
try:
    import pyamg
except:
    pyamg = None


class BaseSeamlessCloner:
    def gradientX(self, img):
        '''
        return the X-axis gradient
        '''
        return img[:, 1:]-img[:, :-1]

    def gradientY(self, img):
        '''
        return the Y-axis gradient
        '''
        return img[1:, :]-img[:-1, ]

    def seamless_clone_single(self, fg, bg, mixed=True, roi=None):
        '''
        clone on a single channel
        '''
        assert fg.shape == bg.shape

        # calculate gradient
        fg_gradient_X, fg_gradient_Y = self.gradientX(fg)[1:-1, :], self.gradientY(fg)[:, 1:-1]
        bg_gradient_X, bg_gradient_Y = self.gradientX(bg)[1:-1, :], self.gradientY(bg)[:, 1:-1]
        here = np.ones((fg.shape[0]-2, fg.shape[1]-2), dtype=bool)

        # whether mixing-gradient mode
        if mixed:
            here &= bg_gradient_X[:, :-1]**2+bg_gradient_Y[:-1,
                                                           :]**2 < fg_gradient_X[:, :-1]**2+fg_gradient_Y[:-1, :]**2
        # whether to use ROI
        if roi is not None:
            here &= roi[1:-1, 1:-1] > 0

        # replace the corresponding gradient
        where = np.where(here)
        bg_gradient_X[where] = fg_gradient_X[where]
        bg_gradient_Y[where] = fg_gradient_Y[where]

        # calculate divergence of gradient
        div = self.gradientX(bg_gradient_X)+self.gradientY(bg_gradient_Y)

        # modify the four sides to simplify the final representation
        div[0, :] -= bg[0, 1:-1]
        div[-1, :] -= bg[-1, 1:-1]
        div[:, 0] -= bg[1:-1, 0]
        div[:, -1] -= bg[1:-1, -1]

        # solve poisson equation
        res = self.poisson_solve(div)

        return res

    def gen_A(self, width, height):
        '''
        generate square matrix A
        '''
        center = -np.ones(width*height)*4
        up = np.ones(width*height)
        down = np.ones(width*height)
        left = np.ones(width*height)
        right = np.ones(width*height)
        left[np.arange(0, height)*width-1] = 0
        right[np.arange(0, height)*width] = 0
        data = [center, up, down, left, right]
        offset = [0, -width, width, -1, 1]
        A = scipy.sparse.dia_matrix(
            (data, offset),
            shape=(height*width,)*2
        )
        return A

    def poisson_solve(self, div):
        '''
        naive solving method, should never be used
        '''
        return np.linalg.solve(self.gen_A(div.shape[1], div.shape[0]).todense(), div.reshape(-1)).reshape(div.shape)

    def seamless_clone(self, fg, bg, point, mixed=False, roi=None):
        '''
        encapsulated poisson clone interface
        '''
        start = time.time()
        self.solve_time = 0

        fg = fg.astype(np.float)/255
        bg = bg.astype(np.float)/255
        fg_h, fg_w, _ = fg.shape
        bg_x, bg_y = point

        # pick out the background fragment as a rectangle
        slice_XY = (slice(bg_y-floor(fg_h/2), bg_y+ceil(fg_h/2)), slice(bg_x-floor(fg_w/2), bg_x+ceil(fg_w/2)))
        slice_XY_reduce = (slice(bg_y-floor(fg_h/2)+1, bg_y+ceil(fg_h/2)-1),
                           slice(bg_x-floor(fg_w/2)+1, bg_x+ceil(fg_w/2)-1))
        bg_t = bg[slice_XY]
        if roi is not None:
            roi = roi[:, :, 0]

        # split BGR channels
        result = np.concatenate([
            self.seamless_clone_single(
                fg[:, :, i], bg_t[:, :, i], mixed=mixed, roi=roi
            )[:, :, None] for i in range(3)
        ], axis=2)

        # assembly back
        bg[slice_XY_reduce] = result

        print("{} done in {:.6f}s, solving taking {:.6f}s".format(
            self.name, time.time()-start, self.solve_time))
        return bg


class CopyPasteCloner(BaseSeamlessCloner):
    name = "CopyPaste solver"

    def seamless_clone_single(self, fg, bg, **kwargs):
        if "roi" in kwargs:
            where = np.where(kwargs["roi"] > 0)
            bg[where] = fg[where]
            return bg[1:-1, 1:-1]
        return fg[1:-1, 1:-1]


class FFTSeamlessCloner(BaseSeamlessCloner):
    name = "FFT solver"

    def poisson_solve(self, div):
        '''
        solve poisson equation using FFT
        '''
        def fft_poisson(f, h):
            n = f.shape[0]
            f_bar = idst2(f)*(2/n+1)**2
            lam = -4*(np.sin(np.arange(1, n+1)*np.pi/(2*(n+1))))**2
            vx, vy = np.meshgrid(range(0, n), range(0, n))
            u_bar = f_bar/(lam[vx]+lam[vy])
            u = dst2(u_bar)*(2/(n+1))**2
            return u

        def dst2(x, axes=(-1, -2)):
            return dst(dst(x, axis=axes[0]), axis=axes[1])

        def idst2(x, axes=(-1, -2)):
            return dst(dst(x, axis=axes[0]), axis=axes[1])

        def dst(x, axis=-1):
            N = x.shape[axis]
            newshape = list(x.shape)
            newshape[axis] = 2*(N+1)
            xsym = np.zeros(newshape)
            slices = [[slice(None), slice(None)]for i in range(3)]
            slices[0][axis] = slice(1, N+1)
            slices[1][axis] = slice(N+2, None)
            slices[2][axis] = slice(None, None, -1)
            slices = [tuple(i)for i in slices]
            xsym[slices[0]] = x
            xsym[slices[1]] = -x[slices[2]]
            DST = fft(xsym, axis=axis)
            return (-(DST.imag)/2)[slices[0]]

        start = time.time()
        result = fft_poisson(div, 1)
        self.solve_time += time.time()-start
        return result

    def seamless_clone(self, fg, bg, point, **kwargs):
        '''
        FFT version single channel cloning
        '''
        height, width, _ = bg.shape
        size = max(height, width)
        scaleX = size/width
        scaleY = size/height
        size = max(fg.shape)
        scaleX = size/fg.shape[1]
        scaleY = size/fg.shape[0]

        # resize both images to square, due to FFT method require input to be a square
        bg = cv2.resize(bg, (int(round(bg.shape[1]*scaleX)), int(round(bg.shape[0]*scaleY))))
        fg = cv2.resize(fg, (int(round(fg.shape[1]*scaleX)), int(round(fg.shape[0]*scaleY))))
        if "roi" in kwargs:
            roi = kwargs["roi"]
            roi = cv2.resize(roi, (int(round(roi.shape[1]*scaleX)), int(round(roi.shape[0]*scaleY))))
            kwargs["roi"] = roi
        point = int(round(point[0]*scaleX)), int(round(point[1]*scaleY))
        return cv2.resize(super().seamless_clone(fg, bg, point, **kwargs), (width, height))


if pyamg:
    class MultigridSeamlessCloner(BaseSeamlessCloner):
        name = "Multigrid solver"

        def poisson_solve(self, div):
            '''
            solve poisson equation using Multigrid method
            '''
            start = time.time()
            x = self.ml.solve(div.reshape(-1), tol=1e-10)
            x = x.reshape(div.shape)
            self.solve_time += time.time()-start
            return x

        def seamless_clone(self, fg, bg, point, **kwargs):
            '''
            Multigrid version single channel cloning
            '''
            self.A = self.gen_A(fg.shape[1]-2, fg.shape[0]-2).tocsr()
            self.ml = pyamg.ruge_stuben_solver(self.A)
            return super().seamless_clone(fg, bg, point, **kwargs)
else:
    class MultigridSeamlessCloner(FFTSeamlessCloner):
        name = "Fake-Multigrid(FFT) solver"

        def seamless_clone(self, fg, bg, point, **kwargs):
            print("[ WARNING !!! ] pyamg is not installed, fallback to FFT solver")
            return super().seamless_clone(fg, bg, point, **kwargs)


def imshow(name, img):
    '''
    show image resize to 1280 pixel on the screen and save it
    '''
    s = 1280/max(img.shape)
    cv2.imshow(name, cv2.resize(img, (int(img.shape[1]*s), int(img.shape[0]*s))))
    cv2.imwrite("images/result/{}.png".format(name).replace(" ", "_"), np.clip(img*255, 0, 255).astype(np.uint8))


def try_all_cloner(name, fg, bg, point, roi=None):
    print("=== demo {} start ===".format(name))
    print("frontground size = {}x{}, background size = {}x{}".format(
        fg.shape[1], fg.shape[0], bg.shape[1], bg.shape[0]))

    blender = CopyPasteCloner()
    result = blender.seamless_clone(fg, bg, point, roi=roi, mixed=False)
    imshow("{} ctrl+cv".format(name), result)
    cv2.waitKey(1)

    blender = MultigridSeamlessCloner()
    result = blender.seamless_clone(fg, bg, point, roi=roi, mixed=False)
    imshow("{} Multigrid".format(name), result)
    cv2.waitKey(1)

    blender = FFTSeamlessCloner()
    result = blender.seamless_clone(fg, bg, point, roi=roi, mixed=False)
    imshow("{} FFT".format(name), result)
    cv2.waitKey(1)

    blender = MultigridSeamlessCloner()
    result = blender.seamless_clone(fg, bg, point, roi=roi, mixed=True)
    imshow("{} Multigrid mixed".format(name), result)
    cv2.waitKey(1)

    blender = FFTSeamlessCloner()
    result = blender.seamless_clone(fg, bg, point, roi=roi, mixed=True)
    imshow("{} FFT mixed".format(name), result)
    cv2.waitKey(1)

    print("=== demo {} end ===".format(name))


def demo_1():
    fg = cv2.imread("images/src/super_fortress.jpg")
    bg = cv2.imread("images/src/sea.jpg")
    roi = cv2.imread("images/src/super_fortress_roi.png")
    fg = cv2.resize(fg, (fg.shape[1]//3*2, fg.shape[0]//3*2))
    roi = cv2.resize(roi, (roi.shape[1]//3*2, roi.shape[0]//3*2))
    # bg = cv2.resize(bg, (bg.shape[1]//3, bg.shape[0]//3))
    point = (300, 600)
    try_all_cloner("super_fortress", fg, bg, point, roi=roi)


def demo_2():
    fg = cv2.imread("images/src/bullfrog.jpg")
    bg = cv2.imread("images/src/river2.jpg")
    roi = cv2.imread("images/src/bullfrog_roi.png")
    fg = cv2.resize(fg, (fg.shape[1]//3, fg.shape[0]//3))
    roi = cv2.resize(roi, (roi.shape[1]//3, roi.shape[0]//3))
    # bg = cv2.resize(bg, (bg.shape[1], bg.shape[0]))
    point = (1700, 670)
    try_all_cloner("bullfrog", fg, bg, point, roi=roi)


def demo_3():
    fg = cv2.imread("images/src/apocalypse.jpg")
    bg = cv2.imread("images/src/big_snowfield.jpg")
    roi = cv2.imread("images/src/apocalypse_roi.png")
    fg = cv2.resize(fg, (fg.shape[1]//2, fg.shape[0]//2))
    roi = cv2.resize(roi, (roi.shape[1]//2, roi.shape[0]//2))
    bg = cv2.resize(bg, (bg.shape[1]//3, bg.shape[0]//3))
    point = (660, 564)
    try_all_cloner("apocalypse", fg, bg, point, roi=roi)


if __name__ == "__main__":
    demo_1()
    demo_2()
    demo_3()

    cv2.waitKey()
