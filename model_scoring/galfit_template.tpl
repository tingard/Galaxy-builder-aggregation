
===============================================================================
# IMAGE and GALFIT CONTROL PARAMETERS
A) $input_fits                                                   # Input data image (FITS file)
B) $output_fits                                                  # Output data image block
C) $sigma_image                                                  # Sigma image name (made from data if blank or "none")
D) $psf_fits                                                     # Input PSF image and (optional) diffusion kernel
E) $psf_fine_sampling                                            # PSF fine sampling factor relative to data
F) $bad_pixel_mask                                               # Bad pixel mask (FITS image or ASCII coord list)
G) $param_constraint_file                                        # File with parameter constraints (ASCII file)
H) $region_xmin    $region_xmax   $region_ymin    $region_ymax   # Image region to fit (xmin xmax ymin ymax)
I) $convolution_box_width    $convolution_box_height             # Size of the convolution box (x y)
J) $photomag_zero                                                # Magnitude photometric zeropoint
K) $plate_scale_dy  $plate_scale_dx                              # Plate scale (dx dy)    [arcsec per pixel]
O) $display_type                                                 # Display type (regular, curses, both)
P) 2                                                             # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps

# INITIAL FITTING PARAMETERS
#
#   For object type, the allowed functions are:
#       nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat,
#       ferrer, powsersic, sky, and isophote.
#
#   Hidden parameters will only appear when they're specified:
#       C0 (diskyness/boxyness),
#       Fn (n=integer, Azimuthal Fourier Modes),
#       R0-R10 (PA rotation, for creating spiral structures).
#
# -----------------------------------------------------------------------------
#   par)    par value(s)    fit toggle(s)    # parameter description
# -----------------------------------------------------------------------------

$object_list

================================================================================
