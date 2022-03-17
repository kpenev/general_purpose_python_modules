"""Define class for combining unrelated MCMC constraints of a parameter."""

import numpy

class CombinedMCMCConstraint:
    """
    Calculate the product of PDFs of independent MCMC constraints on a grid.

    """

    def __init__(self, grid, kernel_scale):
        """Prepare the grid on which combined PDF will be calculated."""

        self._grid = grid
        self._pdf = 1.0
        self._kernel_scale = kernel_scale

    def eval_kernel(self, shifted_scaled):
        """The kernel to apply shifted to zero and scaled to unit width."""

        return numpy.maximum(0.75 * (1.0 - shifted_scaled**2), 0.0)

    def add_samples(self, samples):
        """
        Update the PDF with the given samples.

        If samples is more than 1-D the last axis should iterate over sample
        index (other axes are treated as separate variables).
        """

        nsamples = samples.shape[-1]
        kernel_width = max(self._kernel_scale * nsamples**(-1.0 / 3.0),
                           (self._grid[1:] - self._grid[:-1]).min())
        kernel_width = 0.3
        self._pdf *= 1.0 / kernel_width * numpy.mean(
            self.eval_kernel(
                -(
                    (
                        samples.flatten() - self._grid[:, None]
                    ).reshape(
                        self._grid.shape + samples.shape
                    )
                )
                /
                kernel_width
            ),
            -1
        )

    def __call__(self):
        """Return the PDF on the given grid."""

        return self._pdf / self._pdf.sum(0)
