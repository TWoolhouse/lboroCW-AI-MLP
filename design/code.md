---
title: Multi-Layer Perceptron
subtitle: Code
author: F121584 - Thomas Woolhouse
date: 22/03/2023
numbersections: false
documentclass: article
papersize: A4
fontsize: 11pt
toc: true
geometry: "left=1.5cm,right=1.5cm,top=0.5cm,bottom=0.1cm"
header-includes: |
  \usepackage{pgffor}
  \usepackage{minted}
---

\hrule


\begin{flushleft}
\foreach \file in {PY_PYTHON_FILES} {
  \pagebreak
  \section{\hspace{-1cm}\file}
  \inputminted[linenos,frame=lines,tabsize=4,baselinestretch=1,autogobble,breaklines=true,python3=true,xleftmargin=-0.5cm,xrightmargin=-1cm]{Python}{\file}
}
\foreach \file in {PY_CPP_FILES} {
  \pagebreak
  \section{\hspace{-1cm}\file}
  \inputminted[linenos,frame=lines,tabsize=4,baselinestretch=1,autogobble,breaklines=true,python3=true,xleftmargin=-0.5cm,xrightmargin=-1cm]{Cpp}{\file}
}
\end{flushleft}
