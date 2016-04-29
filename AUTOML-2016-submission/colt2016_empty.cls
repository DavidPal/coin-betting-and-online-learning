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

\LoadClass[wcp]{jmlr_empty}

\jmlrvolume{}
\jmlryear{}

\ifanonsubmission
 \newcommand{\coltauthor}[1]{}
 \author{author names withheld}
 \editor{Under Review for COLT 2016}
 \jmlrworkshop{}
 \renewcommand{\acks}[1]{}
\else
 \newcommand{\coltauthor}[1]{\author{#1}}
 \iffinalsubmission
  \editors{}
  \jmlrworkshop{}
 \fi
\fi



