import cv2
import numpy as np


class NoCircleFound(Exception):
    pass


def _make_circular_mask(height, width, radius):
    """円状のマスクを作成します"""

    x = np.concatenate((np.arange(int(width / 2)), np.arange(int(width / 2))[::-1]))
    y = np.concatenate((np.arange(int(height / 2)), np.arange(int(height / 2))[::-1]))
    circular_mask = (x[np.newaxis, :] - radius) ** 2 + (
        y[:, np.newaxis] - radius
    ) ** 2 < radius**2
    return circular_mask


def _calc_radius_length(img, mode="default"):
    """画像の内容から乳頭画像の半径のピクセル数を求めます"""

    assert mode in ["default", "adaptive"]

    shape = img.shape
    y_center, x_center = int(shape[0] / 2), int(shape[1] / 2)

    if mode == "default":
        radius = shape[0] - y_center
    else:
        radius = max(
            np.sum(img[y_center, x_center:] > 10), np.sum(img[y_center:, x_center] > 10)
        )

    return radius


def _sanitize(fundus_img):
    """画像処理にかける前のformatting処理を行う

    現状、画像のピクセル幅が奇数だとうまく動作しないので、偶数に揃える
    """

    assert fundus_img.ndim == 3

    height, width, _ = fundus_img.shape

    if height % 2 == 1:
        fundus_img = fundus_img[: height - 1, :, :]

    if width % 2 == 1:
        fundus_img = fundus_img[:, : width - 1, :]

    return fundus_img


def _get_foreground_elements(fundus_img, mode="default"):
    """黒い背景以外のピクセルを1dにして返す

    :param mode: default or adaptive. defaultの場合は、写真が欠けず写っていることを想定。それ以外の場合、半径を効率的に推定する
    """
    shape = fundus_img.shape
    y_center, x_center = int(shape[0] / 2), int(shape[1] / 2)

    radius = _calc_radius_length(fundus_img, mode)
    mask = _make_circular_mask(shape[1], shape[0], radius)

    flatten_img = fundus_img.flatten()
    flatten_mask = mask.flatten()

    return flatten_img[flatten_mask]


def _argmax_thresh_interclass_variance(values):
    """クラス間分散を最大にする閾値を1d arrayから求める"""

    candidates = range(np.min(values) + 1, np.max(values))

    best_objective_val = -np.inf
    best_thresh = np.nan

    for th in candidates:
        num_class1 = values[values < th].size
        num_class2 = values[values >= th].size
        mean_class1 = values[values < th].mean()
        mean_class2 = values[values >= th].mean()

        objective_val = num_class1 * num_class2 * (mean_class1 - mean_class2) ** 2
        if objective_val > best_objective_val:
            best_thresh = th
            best_objective_val = objective_val

    if best_thresh == np.nan:
        raise ValueError("thres not found")

    return best_thresh


class FundusImageCropping(object):
    def __init__(
        self,
        img_shape,
        vessel_area_kernel_size=0.0175,  # 25 in train images
        optic_area_kernel_size=0.0527,  # 75 in train images
        blackhat_area_kernel_size=0.0246,  # 35 in train images
        blackhat_preprocess_kernel_size=0.0049,  # 7 in train images
        foreground_calc_method="default",
    ):
        """コンストラクタ

        :param img_shape: 入力画像の次元
        :param vessel_area_kernel_size: 血管検出の結果から乳頭領域を検出する際のハイパーパラメータ。大きくすると乳頭の発見失敗がが起きにくくなるが、haloなどのノイズが含まれやすくなる
        :param optic_area_kernel_size: たたみ込みによる乳頭検出の際のカーネルの半径
        :param blackhat_are_kernel_size: black tophat 変換のカーネルサイズ。大きくするとノイズを含みやすく、小さくすると血管自体を検出しにくくなる
        :param blackhat_preprocess_kernel_size: 血管検出後のmorphology変換に使用するカーネルの大きさ
        """

        img_edge_size = img_shape[0]

        self.vessel_area_kernel_size = int(img_edge_size * vessel_area_kernel_size)
        self.optic_area_kernel_size = int(img_edge_size * optic_area_kernel_size)
        self.blackhat_area_kernel_size = int(img_edge_size * blackhat_area_kernel_size)
        self.blackhat_preprocess_kernel_size = int(
            img_edge_size * blackhat_preprocess_kernel_size
        )
        self.foreground_calc_method = foreground_calc_method

        print(self.__dict__)

        # make circular kernel
        kernel_size = self.optic_area_kernel_size
        kernel = _make_circular_mask(
            kernel_size * 2, kernel_size * 2, kernel_size
        ).astype(np.int64)
        kernel = kernel / np.pi / kernel_size**2

        self.kernel_for_opticdisk = kernel

    def _get_black_tophat_mask(self, img, mode="default"):
        """black tophat transformationによる血管検出

        :param mode: default or adaptive. defaultの場合は、写真が欠けず写っていることを想定。それ以外の場合、半径を効率的に推定する
        """

        def equalize_contrast(img, mode):
            if img.ndim == 3:
                img = img[:, :, 1]

            img_mod = cv2.equalizeHist(img)

            # fill background element with zero
            height, width = img.shape
            radius = _calc_radius_length(img, mode)

            y, x = np.ogrid[
                -int(height / 2) : int(height / 2), -int(width / 2) : int(width / 2)
            ]
            mask = x**2 + y**2 > radius**2
            img_mod[mask] = 0
            return img_mod

        # detect vessels with black tophat transformation
        kernel_size = self.blackhat_area_kernel_size
        btt_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        blackhat = cv2.morphologyEx(
            equalize_contrast(img[:, :, 1], mode), cv2.MORPH_BLACKHAT, btt_kernel
        )

        # make binary vessel mask
        thresh = _argmax_thresh_interclass_variance(
            _get_foreground_elements(blackhat, mode)
        )
        blackhat_mask = (blackhat > thresh).astype(np.uint8)

        morphology_kernel_size = self.blackhat_preprocess_kernel_size
        morphology_kernel = np.ones(
            (morphology_kernel_size, morphology_kernel_size), np.uint8
        )
        # オープニング処理(拡大 -> 縮小による間欠除去)
        blackhat_mask = cv2.morphologyEx(
            blackhat_mask, cv2.MORPH_CLOSE, morphology_kernel
        )
        # オープニング処理(縮小 -> 拡大によるノイズ除去)
        blackhat_mask = cv2.morphologyEx(
            blackhat_mask, cv2.MORPH_OPEN, morphology_kernel
        )

        return blackhat_mask

    def _get_inside_vessel_closure(self, vessel_mask):
        """血管の縁の内部のマスクを作成する"""

        cumsum_top_to_bottom = np.cumsum(vessel_mask, axis=0)
        cumsum_bottom_to_up = np.cumsum(vessel_mask[::-1, :], axis=0)[::-1, :]
        cumsum_left_to_right = np.cumsum(vessel_mask, axis=1)
        cumsum_right_to_left = np.cumsum(vessel_mask[:, ::-1], axis=1)[:, ::-1]

        mask = (cumsum_top_to_bottom * cumsum_bottom_to_up) + (
            cumsum_left_to_right * cumsum_right_to_left
        )
        mask = np.clip(mask, 0, 255).astype(np.uint8)

        morphology_kernel = np.ones(
            (self.vessel_area_kernel_size, self.vessel_area_kernel_size), np.uint8
        )
        # オープニング処理(縮小 -> 拡大によるノイズ除去)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morphology_kernel)
        # オープニング処理(拡大 -> 縮小による間欠除去)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morphology_kernel)

        return mask.astype(bool)

    def _get_brightest_region_center_coordinate(self, img):
        """円状のカーネルをたたみ込んだ時の最大値の座標を求める

        :return: 中心座標(x, y), 半径をタプル形式で
        """

        kernel = self.kernel_for_opticdisk

        conv_img = cv2.filter2D(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            # img[:, :, 0].astype(np.int64),
            -1,
            kernel,
        )
        argmax_idx = np.argmax(conv_img)
        x, y = argmax_idx % img.shape[1], int(argmax_idx / img.shape[1])

        # 実験的に半径をイメージのサイズから決定してみる
        # r = int(min(img.shape[:2]) / 14)
        r = self.optic_area_kernel_size
        return (x, y, r)

        # return (x, y, 100)

    def _get_brightest_houghcircle(self, img):
        """Hough変換で検出された円のうちもっとも輝度の高いものを返す

        :return: 中心座標(x, y), 半径をタプル形式で
        """

        height, width = img.shape[:2]
        min_edge = min(height, width)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ### params for hough transform ###
        method = (
            cv2.HOUGH_GRADIENT
        )  # 現在のところ， CV_HOUGH_GRADIENT メソッドのみが実装されています．基本的には 2段階ハフ変換 で，これについては Yuen90 で述べられています．
        dp = 2  # 画像分解能に対する投票分解能の比率の逆数．例えば， dp=1 の場合は，投票空間は入力画像と同じ分解能をもちます．また dp=2 の場合は，投票空間の幅と高さは半分になります．
        min_dist = int(
            min_edge / 10
        )  # 検出される円の中心同士の最小距離．このパラメータが小さすぎると，正しい円の周辺に別の円が複数誤って検出されることになります．逆に大きすぎると，検出できない円がでてくる可能性があります．
        param1 = 40  # 手法依存の 1 番目のパラメータ． CV_HOUGH_GRADIENT の場合は， Canny() エッジ検出器に渡される2つの閾値の内，大きい方の閾値を表します（小さい閾値は，この値の半分になります）．
        param2 = 20  # 手法依存の 2 番目のパラメータ． CV_HOUGH_GRADIENT の場合は，円の中心を検出する際の投票数の閾値を表します．これが小さくなるほど，より多くの誤検出が起こる可能性があります．より多くの投票を獲得した円が，最初に出力されます．
        min_radius = int(min_edge / 6)
        max_radius = int(min_edge / 3)

        circles = cv2.HoughCircles(
            img_gray,
            method,
            dp,
            min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            raise NoCircleFound()

        max_brightness = 0
        for c in circles[0]:
            circle_filter = np.zeros_like(img)
            cv2.circle(circle_filter, (c[0], c[1]), c[2], (1, 1, 1), -1)
            brightness = np.sum(img * circle_filter[:, :, 0][:, :, np.newaxis]) / float(
                c[2] ** 2
            )
            if brightness > max_brightness:
                max_brightness = brightness
                brightest_circle = np.round(c).astype(np.int)

        return brightest_circle[:3]

    def get_opticdisk_coordinate(self, img, how="conv"):
        """乳頭を中心とした領域の四隅の座標を計算する

        .. Note::
            - 乳頭検出のために、Hough変換による検出と、畳み込みによる白色領域検出とを用意しています。
            - 引数 `how` で指定できます。 `how` には `conv` もしくは `hough` を指定してください
            - `conv`は現状乳頭の半径を上手に推定できませんが、より高速です。`hough`は明示的に円の中心座標・半径を求めることができますが、計算量が大きいです。
        """

        assert how in ["conv", "hough"]

        img = _sanitize(img)

        # detect vessels with black tophat transformation
        blackhat_mask = self._get_black_tophat_mask(
            img, mode=self.foreground_calc_method
        )

        # make binary closure mask by vessels
        closure_mask = self._get_inside_vessel_closure(blackhat_mask)

        img_masked = np.copy(img)
        img_masked[np.invert(closure_mask)] = 0

        if how == "conv":
            x, y, r = self._get_brightest_region_center_coordinate(img_masked)
        else:
            x, y, r = self._get_brightest_houghcircle(img_masked)

        # calculate coordinates to crop
        y_min = y - r * 2
        y_max = y + r * 2
        if y_min < 0:
            y_max = y_max - y_min
            y_min = 0

        if y_max > img.shape[0]:
            y_min = y_min - (y_max - img.shape[0])
            y_max = img.shape[0]

        x_min = x - r * 2
        x_max = x + r * 2
        if x_min < 0:
            x_max = x_max - x_min
            x_min = 0

        if x_max > img.shape[1]:
            x_min = x_min - (x_max - img.shape[1])
            x_max = img.shape[1]

        return (x_min, x_max, y_min, y_max)

    def get_opticdisk(self, img, resize_to=None, how="conv"):
        """乳頭を中心とした切り出し画像を返す

        .. Note::
            - 乳頭検出のために、Hough変換による検出と、畳み込みによる白色領域検出とを用意しています。
            - 引数 `how` で指定できます。 `how` には `conv` もしくは `hough` を指定してください
            - `conv`は現状乳頭の半径を上手に推定できませんが、より高速です。`hough`は明示的に円の中心座標・半径を求めることができますが、計算量が大きいです。
        """

        x_min, x_max, y_min, y_max = self.get_opticdisk_coordinate(img, how)

        img_cropped = img[y_min:y_max, x_min:x_max]
        if resize_to is not None:
            img_cropped = cv2.resize(img_cropped, (resize_to, resize_to))

        return img_cropped

    def get_opticdisk_coordinate_with_macula(self, img, resize_to=None, how="conv"):
        """乳頭を中心とし、黄斑を含めた領域の四隅の座標を計算する
        .. Note::
            - 乳頭検出のために、Hough変換による検出と、畳み込みによる白色領域検出とを用意しています。
            - 引数 `how` で指定できます。 `how` には `conv` もしくは `hough` を指定してください
            - `conv`は現状乳頭の半径を上手に推定できませんが、より高速です。`hough`は明示的に円の中心座標・半径を求めることができますが、計算量が大きいです。
        """

        assert how in ["conv", "hough"]

        img = _sanitize(img)

        # detect vessels with black tophat transformation
        blackhat_mask = self._get_black_tophat_mask(
            img, mode=self.foreground_calc_method
        )

        # make binary closure mask by vessels
        closure_mask = self._get_inside_vessel_closure(blackhat_mask)

        img_masked = np.copy(img)
        img_masked[np.invert(closure_mask)] = 0

        if how == "conv":
            x, y, r = self._get_brightest_region_center_coordinate(img_masked)
        else:
            x, y, r = self._get_brightest_houghcircle(img_masked)

        # judge whether the image is left eye's or right
        # is_left_eye = blackhat_mask[:, :int(blackhat_mask.shape[1]/2)] .sum() < blackhat_mask[:, int(blackhat_mask.shape[1]/2):] .sum()
        is_left_eye = x > blackhat_mask.shape[1] / 2
        if is_left_eye:
            x_min = x - r * 6
            x_max = x + r * 2
        else:
            x_min = x - r * 2
            x_max = x + r * 6

        y_min = y - r * 4
        y_max = y + r * 4

        if y_min < 0:
            y_max = y_max - y_min
            y_min = 0

        if y_max > img.shape[0]:
            y_min = y_min - (y_max - img.shape[0])
            y_max = img.shape[0]

        if x_min < 0:
            x_max = x_max - x_min
            x_min = 0

        if x_max > img.shape[1]:
            x_min = x_min - (x_max - img.shape[1])
            x_max = img.shape[1]

        return (x_min, x_max, y_min, y_max)

    def get_opticdisk_with_macula(self, img, resize_to=None, how="conv"):
        """乳頭を中心とし、黄斑を含めた切り出し画像を返す
        .. Note::
            - 乳頭検出のために、Hough変換による検出と、畳み込みによる白色領域検出とを用意しています。
            - 引数 `how` で指定できます。 `how` には `conv` もしくは `hough` を指定してください
            - `conv`は現状乳頭の半径を上手に推定できませんが、より高速です。`hough`は明示的に円の中心座標・半径を求めることができますが、計算量が大きいです。
        """
        x_min, x_max, y_min, y_max = self.get_opticdisk_coordinate_with_macula(img, how)

        img_cropped = img[y_min:y_max, x_min:x_max]
        if resize_to is not None:
            img_cropped = cv2.resize(img_cropped, (resize_to, resize_to))

        return img_cropped
