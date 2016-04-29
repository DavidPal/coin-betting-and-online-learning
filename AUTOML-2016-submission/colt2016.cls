\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{colt2016}[2016/01/11 v1.0 COLT 2016 Proceedings]

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[wcp]{jmlr}

\jmlrvolume{vol 49}
\jmlryear{2016}

\ifanonsubmission
 \newcommand{\coltauthor}[1]{}
 \author{author names withheld}
 \editor{Under Review for COLT 2016}
 \jmlrworkshop{29th Annual Conference on Learning Theory}
 \renewcommand{\acks}[1]{}
\else
 \newcommand{\coltauthor}[1]{\author{#1}}
 \iffinalsubmission
  \editors{Vitaly Feldman, Alexander Rakhlin and Ohad Shamir}
  \jmlrworkshop{29th Annual Conference on Learning Theory}
 \fi
\fi



