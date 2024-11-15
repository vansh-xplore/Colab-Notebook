{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMF1nfHpblNj9OW2BiLCeRR",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/vansh-xplore/Colab-Notebook/blob/main/all_program.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**This is a Code of nth root of a complex number**\n",
        "\n",
        "---\n",
        "The nth root of a complex number z=r(cosθ+isinθ) is any complex number 𝑤 sunch that w^n=z."
      ],
      "metadata": {
        "id": "A-QQt_ATUJTD"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-jSxkzn9X598"
      },
      "outputs": [],
      "source": [
        "import cmath\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def nth_root(z,n):\n",
        "    r,theta=cmath.polar(z)\n",
        "    omega=r**(1/n)\n",
        "    roots=[]\n",
        "    argument=[]\n",
        "\n",
        "    for i in range(n):\n",
        "        # Argument of each root\n",
        "        phi=(theta+2*np.pi*i)/n\n",
        "\n",
        "        root_magintude=omega\n",
        "        root_argument=phi\n",
        "\n",
        "        # Root in Cartesian form\n",
        "        root_cartesain=cmath.rect(root_magintude,root_argument)\n",
        "        roots.append(root_cartesain)\n",
        "        argument.append(root_argument)\n",
        "\n",
        "        print(f'Root {i}:')\n",
        "        print(f'In Polar form: Magnitude= {root_magintude}, Angle={root_argument} radians')\n",
        "        print(f'In cartesian form: Root={root_cartesain}')\n",
        "\n",
        "    return roots, argument, omega\n",
        "\n",
        "# Parameters\n",
        "z=3+4j\n",
        "n=5\n",
        "\n",
        "# Radius is root_magnitude (Magnitude)\n",
        "roots,argument, radius=nth_root(z,n)\n",
        "\n",
        "print('')\n",
        "\n",
        "plt.figure(figsize=(6,6))\n",
        "\n",
        "for i in range(n):\n",
        "    print(f'Argument of {i}th root in degrees is ({np.degrees(argument[i])})')\n",
        "\n",
        "    root=roots[i]\n",
        "    # Line from origin to root\n",
        "    plt.plot([0,root.real],[0,root.imag])\n",
        "\n",
        "    arg_deg=np.degrees(argument[i])\n",
        "    plt.scatter(root.real,root.imag,label=f'{i}-th root')\n",
        "    plt.text(root.real,root.imag,f'θ={np.degrees(argument[i]):.2f}°')\n",
        "\n",
        "angles=np.linspace(0,2*np.pi,100)\n",
        "circle_x=radius*np.cos(angles)\n",
        "circle_y=radius*np.sin(angles)\n",
        "\n",
        "plt.xlim(-2,2)\n",
        "plt.ylim(-2,2)\n",
        "\n",
        "#Plot the circle\n",
        "plt.plot(circle_x,circle_y,color='green')\n",
        "\n",
        "plt.axhline(0,color='black',linewidth=0.5,linestyle='dashdot')\n",
        "plt.axvline(0,color='black',linewidth=0.5,linestyle='dashdot')\n",
        "\n",
        "plt.grid(linewidth=0.5,linestyle='dashdot')\n",
        "\n",
        "plt.title(f'All {n}th Roots of {z}')\n",
        "plt.xlabel('Real Part')\n",
        "plt.ylabel('Imaginary Part')\n",
        "plt.legend()\n",
        "\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**This is a code of Translation of a complex Number.**\n",
        "\n",
        "---\n",
        "The translation of a complex number refers to adding another complex number to it, resulting in a shift or movement in the complex plane without changing its orientation or rotation."
      ],
      "metadata": {
        "id": "gZZGAFGCUHFn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Original square vertices as complex numbers\n",
        "v1 = 0 + 0j\n",
        "v2 = 1 + 0j\n",
        "v3 = 1 + 1j\n",
        "v4 = 0 + 1j\n",
        "\n",
        "# Translation vector as a complex number\n",
        "translation = 2 + 1j\n",
        "\n",
        "# Apply the translation to each vertex\n",
        "v1_translated = v1 + translation\n",
        "v2_translated = v2 + translation\n",
        "v3_translated = v3 + translation\n",
        "v4_translated = v4 + translation\n",
        "\n",
        "# Calculate the side length of original square\n",
        "side_length=abs(v2-v1)\n",
        "\n",
        "# Calculate the side length of translated square\n",
        "side_length_translated=abs(v2_translated-v1_translated)\n",
        "\n",
        "# Calculate the area of original and translated square\n",
        "area_original=side_length**2\n",
        "area_translated=side_length_translated**2\n",
        "\n",
        "# Check the areas are the same\n",
        "if area_original == area_translated:\n",
        "    print(f'The area are the same. Area: {area_original}')\n",
        "else:\n",
        "    print('The area are different')\n",
        "\n",
        "# Plotting the original square\n",
        "plt.plot([v1.real, v2.real], [v1.imag, v2.imag], 'bo-')\n",
        "plt.plot([v2.real, v3.real], [v2.imag, v3.imag], 'bo-')\n",
        "plt.plot([v3.real, v4.real], [v3.imag, v4.imag], 'bo-')\n",
        "plt.plot([v4.real, v1.real], [v4.imag, v1.imag], 'bo-', label='Original Square')\n",
        "\n",
        "# Plotting the translated square\n",
        "plt.plot([v1_translated.real, v2_translated.real], [v1_translated.imag, v2_translated.imag], 'ro-')\n",
        "plt.plot([v2_translated.real, v3_translated.real], [v2_translated.imag, v3_translated.imag], 'ro-')\n",
        "plt.plot([v3_translated.real, v4_translated.real], [v3_translated.imag, v4_translated.imag], 'ro-')\n",
        "plt.plot([v4_translated.real, v1_translated.real], [v4_translated.imag, v1_translated.imag], 'ro-', label='Translated Square')\n",
        "\n",
        "# Plotting grey lines connecting original and translated vertices\n",
        "plt.plot([v1.real, v1_translated.real], [v1.imag, v1_translated.imag], color='grey')\n",
        "plt.plot([v2.real, v2_translated.real], [v2.imag, v2_translated.imag], color='grey')\n",
        "plt.plot([v3.real, v3_translated.real], [v3.imag, v3_translated.imag], color='grey')\n",
        "plt.plot([v4.real, v4_translated.real], [v4.imag, v4_translated.imag], color='grey')\n",
        "\n",
        "# Configure the plot\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "plt.xlabel('Real')\n",
        "plt.ylabel('Imaginary')\n",
        "plt.title('Translation of a Square using Complex Numbers')\n",
        "plt.axis('equal')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "v909CTO2YPNd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**This is a code of rotation of a Complex Number**\n",
        "\n",
        "---\n",
        "\n",
        "The rotation of a complex number involves multiplying it by another complex number, typically one of modulus 1, to change its angle without altering its magnitude."
      ],
      "metadata": {
        "id": "xk-5hwktXHQT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "#Aim- Transformation of complex numbers as 2-D vectors by rotation\n",
        "\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as git\n",
        "import cmath\n",
        "\n",
        "#Define the function\n",
        "def rotate(z,angle):\n",
        "    return z*np.exp(1j*cmath.pi*angle/180)\n",
        "z=1+1j\n",
        "angle=45\n",
        "rotated_z=rotate(z,angle)\n",
        "\n",
        "#Find the length of these two lines\n",
        "original_length = abs(z)\n",
        "rotated_length = abs(rotated_z)\n",
        "\n",
        "# Check the length of these two lines in same or not\n",
        "if original_length == rotated_length:\n",
        "    print(f'length of orig_comp={round(original_length,4)}, length of rota_comp={round(rotated_length,4)}')\n",
        "    print('Both length is same')\n",
        "else:\n",
        "    print('The length of these lines is not same')\n",
        "\n",
        "\n",
        "'''\n",
        "The angle between these two lines,\n",
        "theta=arccos(real(original)*real(rotate)+imag(original)*imag(rotate)/|original|*|rotate|)*180/pi {degrees}\n",
        "'''\n",
        "\n",
        "# Find the angle between these two lines\n",
        "dot_pro=z.imag*rotated_z.imag+z.real*rotated_z.real\n",
        "cos_theta=dot_pro/(original_length*rotated_length)\n",
        "angle_bet=np.arccos(cos_theta)*180/np.pi # Make the angle in degress\n",
        "print(f'The angle between the original and rotated lines is {round(angle_bet,2)} degrees')\n",
        "\n",
        "\n",
        "# Plotting the Lines\n",
        "git.plot([0,z.real],[0,z.imag],label='Original')\n",
        "git.plot([0,rotated_z.real],[0,rotated_z.imag],label='Rotated')\n",
        "\n",
        "# PLotting the arc\n",
        "angles=np.linspace(0,angle,100)\n",
        "arc_points=z*np.exp(1j*np.pi*angles/180)\n",
        "git.plot(arc_points.real,arc_points.imag)\n",
        "\n",
        "git.gca().set_aspect('equal', adjustable='box')\n",
        "git.grid()\n",
        "git.legend()\n",
        "git.xlabel('Real Part')\n",
        "git.ylabel('Imaginary Part')\n",
        "git.title('Rotation of a complex number')\n",
        "git.legend()\n",
        "git.show()\n"
      ],
      "metadata": {
        "id": "tnZos7kGYUWs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**This is a code of Magnification of Complex Number**\n",
        "\n",
        "---\n",
        "The magnification (or scaling) of a complex number involves multiplying it by a real scalar, which changes its distance from the origin without altering its angle."
      ],
      "metadata": {
        "id": "Iqd_Vm5wXccl"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "def magnification(z,m):\n",
        "    z_magnified = z*m\n",
        "    return z_magnified\n",
        "z = 3+4j\n",
        "m = 5\n",
        "z_magnified = magnification(z,m)\n",
        "\n",
        "# Plot the original and magnified complex numbers\n",
        "r_original = np.abs(z)\n",
        "theta_original = np.angle(z)\n",
        "r_magnified = np.abs(z_magnified)\n",
        "theta_magnified = np.angle(z_magnified)\n",
        "\n",
        "# Plotting in polar coordinates\n",
        "plt.figure(figsize=(10, 5))\n",
        "\n",
        "# Original complex number\n",
        "plt.subplot(1, 2, 1)\n",
        "plt.plot([0, theta_original], [0, r_original], 'bo-', label='Original')\n",
        "plt.title('Original Complex Number')\n",
        "plt.grid()\n",
        "plt.xlabel('Real Axis')\n",
        "plt.ylabel('Imaginary Axes')\n",
        "\n",
        "# Magnified complex number\n",
        "plt.subplot(1, 2, 2)\n",
        "plt.plot([0, theta_magnified], [0, r_magnified], 'ro-', label='Magnified')\n",
        "plt.title('Magnified Complex Number')\n",
        "plt.grid()\n",
        "plt.xlabel('Real Axis')\n",
        "plt.ylabel('Imaginary Axes')\n",
        "\n",
        "plt.suptitle('Magnification of complex number')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
      ],
      "metadata": {
        "id": "kJX5ug39YYTb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**This is a code of Squaring of a complex Number**\n",
        "\n",
        "---\n",
        "The squaring of a complex number involves multiplying the complex number by itself, resulting in a new complex number. This operation impacts both the magnitude (or modulus) and the angle (or argument) of the original complex number.\n"
      ],
      "metadata": {
        "id": "gcL0xwPaYDD-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Define the circular arc in polar coordinates\n",
        "theta = np.linspace(0, np.pi / 2, 100)  # Angle range\n",
        "r = 1  # Radius of the circular arc\n",
        "\n",
        "# Convert to complex numbers (z = r * e^(i*theta))\n",
        "z = r * np.exp(1j * theta)\n",
        "\n",
        "# Apply the z^2 mapping\n",
        "w = z**2\n",
        "\n",
        "# Plotting in polar coordinates\n",
        "fig,axes=plt.subplots(1,2,figsize=(12,6))\n",
        "# Original circular arc\n",
        "axes[0].plot(z.real,z.imag, label='θ∈(0,π/2)', color='blue')\n",
        "axes[0].set_title('Original Circular Arc')\n",
        "axes[0].set_xlabel('Real Part')\n",
        "axes[0].set_ylabel('Imaginary Part')\n",
        "axes[0].grid(True)\n",
        "axes[0].legend()\n",
        "\n",
        "# Image of the circular arc under z^2 mapping\n",
        "axes[1].plot(w.real,w.imag, label='θ∈(0,π)', color='red')\n",
        "axes[1].set_title('Image under z^2 Mapping')\n",
        "axes[1].set_xlabel('Real Part')\n",
        "axes[1].set_ylabel('Imaginary Part')\n",
        "axes[1].grid(True)\n",
        "axes[1].legend()\n",
        "\n",
        "plt.suptitle('Mapping of f(z)=z**2 for θ∈(0,π/2)')\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
      ],
      "metadata": {
        "id": "EnoN4un9YdPr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**This is a code of linear mapping of a Complex functions from one plane to another**\n",
        "\n",
        "---\n",
        "In complex analysis, linear mappings (or linear transformations) of a complex function involve transformations that can be represented by functions of the form f(z)=az+b, where a and b are complex constants, and z is a complex variable. These mappings have fundamental properties that make them powerful tools for transforming complex planes and are widely used in geometry and complex function theory."
      ],
      "metadata": {
        "id": "qnS1UVEFYVVf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "imag_values=np.linspace(-15,15,100)\n",
        "k=[-2,-1,0,1,2]\n",
        "for i in k:\n",
        "    plt.subplot(1,2,1)\n",
        "    z=k[i]+1j*imag_values\n",
        "    plt.plot(z.real,z.imag,label=f'for z={i}')\n",
        "    plt.xlabel('Real Part')\n",
        "    plt.ylabel('Imaginary Part')\n",
        "    plt.title('Verticals lines: (f(z)=k)')\n",
        "    plt.grid()\n",
        "    plt.legend()\n",
        "\n",
        "    plt.subplot(1,2,2)\n",
        "    u=k[i]**2-imag_values**2\n",
        "    v=2*k[i]*imag_values\n",
        "    plt.plot(u,v,label=f'for z={i}')\n",
        "    plt.xlabel('Real Part')\n",
        "    plt.ylabel('Imaginary Part')\n",
        "    plt.title('Parabola for (f(z)=k)')\n",
        "    plt.grid()\n",
        "    plt.legend()\n",
        "\n",
        "plt.suptitle('Mapping of z=k from z to w plane')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "krEh3CW1Yvsk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**This is a code of linear mapping of a Complex functions from one plane to another**\n",
        "\n",
        "---\n",
        "The exponential mapping in complex analysis involves the function f(z) = e^z,  where z is a complex variable. This transformation, also known as the complex exponential function, maps the complex plane in a way that has unique properties and applications. The exponential map transforms lines and regions in the complex plane into curves and shapes that can differ significantly from their original forms.\n"
      ],
      "metadata": {
        "id": "HhtUKxMUZQ0n"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "imag_values=np.linspace(-15,15,100)\n",
        "k=[-2,-1,0,1,2]\n",
        "for i in k:\n",
        "    plt.subplot(1,2,1)\n",
        "    z=k[i]+1j*imag_values\n",
        "    plt.plot(z.real,z.imag,label=f'for z={i}')\n",
        "    plt.xlabel('Real Part')\n",
        "    plt.ylabel('Imaginary Part')\n",
        "    plt.title('Verticals lines: (f(z)=k)')\n",
        "    plt.grid()\n",
        "    plt.legend()\n",
        "\n",
        "    plt.subplot(1,2,2)\n",
        "    u=np.exp(k[i])*np.cos(imag_values)\n",
        "    v=np.exp(k[i])*np.sin(imag_values)\n",
        "    plt.plot(u,v,label=f'for z={i}')\n",
        "    plt.xlabel('Real Part')\n",
        "    plt.ylabel('Imaginary Part')\n",
        "    plt.title('Parabola for (f(z)=k)')\n",
        "    plt.grid()\n",
        "    plt.legend()\n",
        "\n",
        "plt.gca().set_aspect('equal',adjustable='box')\n",
        "plt.suptitle('Mapping of z=k from z to w plane')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "94DBtvACYyZV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**This is a code of Guass_quadrature Method.**\n",
        "\n",
        "---\n",
        "The Gauss Quadrature method is a numerical integration technique used to approximate the definite integral of a function. It is particularly efficient for integrating functions that can be expressed as polynomials or approximated by polynomials, achieving high accuracy with fewer sample points compared to other methods like the trapezoidal or Simpson's rule. The method utilizes special sample points (nodes) and associated weights, chosen to optimize accuracy."
      ],
      "metadata": {
        "id": "626eBT81Zt6f"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def func(x):\n",
        "    return np.exp(x)\n",
        "\n",
        "a=0\n",
        "b=2\n",
        "n=2\n",
        "\n",
        "# Plot the function over the limits\n",
        "x=np.linspace(a,b,100)\n",
        "y=func(x)\n",
        "\n",
        "# Plot the function\n",
        "plt.plot(x,y,color='blue',label='exp(x)')\n",
        "plt.fill_between(x,y,color='lightgreen',alpha=0.3)\n",
        "\n",
        "if n == 1:\n",
        "    weight=b-a\n",
        "    node=(b+a)/2\n",
        "    integral=weight*func(node)\n",
        "\n",
        "    # Plot the node\n",
        "    plt.scatter(node,func(node),color='red',label=f'Node 1: x={node:.4}')\n",
        "\n",
        "    print(f'The value of integration for one point is {integral:.4}')\n",
        "\n",
        "elif n == 2:\n",
        "    weight=(b-a)/2\n",
        "    node1=weight*(-1/(3**0.5))+(b+a)/2\n",
        "    node2=weight*(1/(3**0.5))+(b+a)/2\n",
        "    integral=weight*func(node1)+weight*func(node2)\n",
        "\n",
        "    # Plot the nodes\n",
        "    plt.scatter(node1,func(node1),color='grey',label=f'Node 1: x={node1:.4}')\n",
        "    plt.scatter(node2,func(node2),color='red',label=f'Node 2: x={node2:.4}')\n",
        "\n",
        "    n1=np.linspace(node1,node2,100)\n",
        "    n2=func(n1)\n",
        "    plt.fill_between(n1,n2,color='lightblue')\n",
        "\n",
        "    print(f'The value of integration for two point is {integral:.4}')\n",
        "\n",
        "else:\n",
        "    print('Try for one or two point gauss quadrature')\n",
        "\n",
        "plt.axhline(0,color='grey')\n",
        "plt.axvline(0,color='grey')\n",
        "plt.text(0.75,0.75,'''\n",
        " Gauss Quadrature\n",
        "Integration Method\n",
        "''')\n",
        "\n",
        "plt.title(f'Integration of exp(x) from {a} to {b} for {n} point')\n",
        "plt.xlabel('x')\n",
        "plt.ylabel('y')\n",
        "plt.legend()\n",
        "plt.grid(color='grey',linewidth=0.5,linestyle='dashdot')\n",
        "plt.show()\n"
      ],
      "metadata": {
        "id": "nyiap0H3Y5qj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**This is a code of Guass_quadrature Method.**\n",
        "\n",
        "---\n",
        "Gauss-Laguerre Quadrature is a numerical integration method used to approximate integrals. This method is a special case of the more general Gauss quadrature techniques, tailored specifically for functions involving an exponential decay term, e^-x.\n"
      ],
      "metadata": {
        "id": "Klxj7D5waHZR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "import numpy as np\n",
        "from scipy.integrate import fixed_quad\n",
        "\n",
        "def func_original(x):\n",
        "    return np.cos(x) * np.exp(-x)\n",
        "\n",
        "def func_transform(t):\n",
        "    s = (1 + t) / (1 - t)\n",
        "    derivative = 2 / (1 - t) ** 2\n",
        "    return func_original(s) * derivative\n",
        "\n",
        "def func_laguerre(x):\n",
        "    return np.cos(x)\n",
        "\n",
        "n = int(input('Enter the number of points (n) for Laguerre (positive integer): '))\n",
        "\n",
        "# Loop starts from 1 to n to avoid using 0, which is invalid for integration methods\n",
        "for i in range(3, n + 1):\n",
        "    # Using fixed_quad with n points\n",
        "    result_legendre = fixed_quad(func_transform, -1, 1, n=i)[0]\n",
        "\n",
        "    # Using i points for Laguerre quadrature\n",
        "    nodes, weights = np.polynomial.laguerre.laggauss(i)\n",
        "    result_laguerre = np.sum(weights * func_laguerre(nodes))\n",
        "\n",
        "    error = abs(result_legendre - result_laguerre)\n",
        "\n",
        "    print(f'The value of the integral for Gauss-Legendre in n={i} is {result_legendre}')\n",
        "    print(f'The value of the integral for Gauss-Laguerre in n={i} is {result_laguerre}')\n",
        "    print(f'The error between these values of the integral is {error}')\n",
        "    print('')\n"
      ],
      "metadata": {
        "id": "s77xZBP-Y_n1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**This is a code of Guass_quadrature Method.**\n",
        "\n",
        "---\n",
        "Gauss-Hermite Quadrature is a numerical integration method designed specifically for approximating integrals. This method is particularly useful for integrals involving the weight function e^-x(^2).\n"
      ],
      "metadata": {
        "id": "eadqnS-4alvn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "import numpy as np\n",
        "from scipy.integrate import fixed_quad\n",
        "\n",
        "def func_original(x):\n",
        "    return np.cos(x) * np.exp(-x**2)\n",
        "\n",
        "def func_transform(t):\n",
        "    s = np.tan((np.pi/2) * t)  # Transforming variable from [-1, 1] to (-inf, inf)\n",
        "    derivative = (np.pi/2) / (np.cos((np.pi/2) * t) ** 2)  # Derivative of the transformation\n",
        "    return func_original(s) * derivative  # Return transformed function\n",
        "\n",
        "def func_hermite(x):\n",
        "    return np.cos(x)  # For Hermite quadrature, use the original function\n",
        "\n",
        "n = int(input('Enter the value of points (n) for Hermite: '))\n",
        "\n",
        "# Adjust the range of i to start from 3\n",
        "for i in range(3, n + 1):\n",
        "    # Calculate integral using Gauss-Legendre quadrature\n",
        "    result_legendre = fixed_quad(func_transform, -1, 1, n=i)[0]\n",
        "\n",
        "    # Calculate integral using Gauss-Hermite quadrature\n",
        "    nodes, weights = np.polynomial.hermite.hermgauss(i)\n",
        "    result_hermite = np.sum(weights * func_hermite(nodes))\n",
        "\n",
        "    # Calculate the error between the two results\n",
        "    error = abs(result_legendre - result_hermite)\n",
        "\n",
        "    # Print results\n",
        "    print(f'The value of the integral for Gauss-Legendre in n={i} is {result_legendre}')\n",
        "    print(f'The value of the integral for Gauss-Hermite in n={i} is {result_hermite}')\n",
        "    print(f'The error between these values of the integral is {error}')\n",
        "    print('')\n"
      ],
      "metadata": {
        "id": "C953HvNNZGoH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**This is a code of Euler's Method**\n",
        "\n",
        "---\n",
        "Euler's Method is a simple, first-order numerical technique for solving ordinary differential equations (ODEs). It provides an approximation to the solution of an ODE by iteratively stepping forward from an initial condition.\n"
      ],
      "metadata": {
        "id": "R7MU6TIJbBrr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def func_slope(x,y):\n",
        "    return (-2*y)+(x**3)*np.exp(-2*x)\n",
        "\n",
        "def func_euler(func,x0,y0,x_end,h):\n",
        "    x_values=np.arange(x0,x_end,h)\n",
        "    y_values=[y0]\n",
        "\n",
        "    for i in range(len(x_values)):\n",
        "        y_new=y_values[i]+h*func(x_values[i],y_values[i])\n",
        "        y_values.append(y_new)\n",
        "\n",
        "    return x_values,y_values\n",
        "\n",
        "x0=0\n",
        "y0=1\n",
        "x_end=0.1\n",
        "h=0.1\n",
        "\n",
        "x_values,y_values=func_euler(func_slope,x0,y0,x_end,h)\n",
        "\n",
        "euler_value=y_values[-1]\n",
        "print(f\"Euler's value in h={h} is {euler_value:.9}\")\n"
      ],
      "metadata": {
        "id": "Dw6E4nuaZJ8e"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}