\documentclass{article}

\usepackage[final]{neurips_2019}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{lipsum}

\newcommand{\note}[1]{\textcolor{blue}{{#1}}}

\title{
  Backpack Model Performance on Monolingual and Multilingual Lexical Relatedness Tasks \\
  \vspace{1em}
  \small{\normalfont Columbia COMS4705 Project Milestone} \\
  \small{\normalfont \textbf{Keywords:} \textit{backpack language models, lexical relatedness tasks, multilingual data}}
}

\author{
  Jenny Ries \\
  Department of Computer Science \\
  Columbia University \\
  \texttt{jr3954@columbia.edu} \\
  \And
  Kavya Venkatesh \\
  Department of Computer Science \\
  Columbia University \\
  \texttt{kv2458@columbia.edu} \\
  % Examples of more authors
%   \And
%   Name \\
%   Department of Computer Science \\
%   Columbia University \\
%   \texttt{$<$UNI$>$@columbia.edu}
}

\begin{document}

\maketitle

\begin{center}
    \note{This template is built on NeurIPS 2019 template\footnote{\url{https://www.overleaf.com/latex/templates/neurips-2019/tprktwxmqmgk}} and adapted from Stanford CS224N Natural Language Processing with Deep Learning 
    }
\end{center}

\begin{abstract}
  Your abstract should motivate the problem, describe your goals, and highlight your main findings. Given that your project is still in progress, it is okay if your findings are what you are still working on.

  We seek to build on the monolingual work of Hewitt et al. and Hao et al. on Backpack Language Models, to bring Backpacking into the multilingual space. 
  Backpack Language Models represent words as weighted combinations of multiple sense vectors, enabling interpretable lexical representations. While previous work has focused on monolingual English settings, we extend Backpack models to multilingual French-English data. We implement a complete Backpack architecture from scratch and develop comprehensive evaluation metrics for assessing multilingual lexical relatedness. Our implementation includes: (1) training Backpack models from scratch on the Europarl parallel corpus, (2) a framework for finetuning pretrained Backpack models, and (3) evaluation tools for word-level, sentence-level, and sense-level analysis across languages. We evaluate cross-lingual word and sentence similarity, sense vector alignment, and multilingual sense representations. Our codebase provides a foundation for understanding how Backpack models capture multilingual lexical relationships through their sense vector representations.
\end{abstract}


\section{Key Information to include}
\begin{itemize}
    \item Mentor: John Hewitt
    \item External collaborators (if no, indicate ``No''): No
    \item Sharing project (if no, indicate ``No''):  No
\end{itemize}

% {\color{red} This template does not contain the full instruction set for this assignment; please refer back to the milestone instructions PDF.}

\section{Approach}
This section details your approach to the problem. 
\begin{itemize}
    \item Our project has two main parts: fine-tuning a Backpack model pretrained on the OpenWebText English corpus on the EuroParl parallel French-English corpus, and training our own implementation of a Backpack Language Model from scratch on the EuroParl corpus.
    \item From here, we will assess performance on lexical relatedness tasks.
    \item Our project draws on work by Hao et al. and Hewitt et al.  For the finetuning section, our baseline is the performance of Hewitt et al.'s Small Backpack Language Model on English lexical relatedness tasks (2023). We will assess if multilingual performance is comparable to monolingual performance.
    \item For the Backpack model trained from scratch on the French-English EuroParl corpus,  we use a similar approach to Hao et al., comparing our implementation of the Backpack model with a transformer model with the same number of layers  and the same parameters, embedding size and tokenizer, also trained on French-English EuroParl (2023).



    \item Please be specific when describing your main approaches. You may want to include key equations and figures (though it is fine if you want to defer creating time-consuming figures until the final report).
    \item Describe your baselines. Depending on space constraints and how standard your baseline is, you might do this in detail or simply refer to other papers for details. 
    % Default project teams can do the latter when describing the provided baseline model.
    \item If any part of your approach is original, make it clear. For models and techniques that are not yours, provide references.
    \item 
    \item If you are using any code that you did not write yourself, make it clear and provide a reference or link. 
    When describing something you coded yourself, make it clear.
\end{itemize} 


\section{Experiments}
This section is expected to contain the following.
\begin{itemize}
    \item \textbf{Data}: Describe the dataset(s) you are using along with references. Make sure the task associated with the dataset is clearly described.
    \item EuroParl French-English corpus
    \item \textbf{Evaluation method}: Describe the evaluation metric(s) you used, plus any other details necessary to understand your evaluation.
    \item Lexical Relatedness task:
    \item \textbf{Experimental details}: Please explain how you ran your experiments (e.g. model configurations, learning rate, training time, etc.).
    \item \textbf{Results}: Report the quantitative results that you have so far. Use a table or plot to compare multiple results and compare against your baselines.
\end{itemize}


\section{Future work}
We have yet to evaluate the Backpack Language Model we are training from scratch on the lexical relateness tasks.

\bibliographystyle{unsrt}
\bibliography{references}
\item 
J. Hewitt, J. Thickstun, C. Manning, and P. Liang. ``Backpack language models." In \textit{Proceedings of the Association for Computational Linguistics}. Association for Computational Linguistics. 2023.

\end{document}
