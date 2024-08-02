# B. Contributors to this Work {.unnumbered}

<!-- Contributors to this work -->

## The Working Group {.unnumbered}

The thesis is part of a larger team project and therefore has to include a statement detailing the proportion of work done by me.
I generally described my own work in the thesis, leaving out other parts of the project to be published later in a journal.
The notable exceptions are:

Below I list the contributions of other members of our group:

* The data filtration methodology of the "Reference Set", which was developed by Jiří Černý and the staff of his laboratory.
* The basepair selection criteria were checked and adjusted by Lada Biedermannová.
* We devised the new basepair parameters in brainstorming sessions with Jiří, Bohdan and Lada and were vetted by the entire group.
* Valuable feedback provided by all members of the working group has had a noticeable impact on my work, mainly on the basepair parameters and the web application.
* The work described in [sec. Conclusions --- Ongoing Work @sec:concl-future] is primarily being performed by others while I am writing this text.

The working group is coordinated by B. Schneider, and the other active members are H. M. Berman, L. Biedermannová, J. Černý, R. Joosten, M. Magnus, E. Westhof, and C. Zirbel.

## Use of Large Language Models {.unnumbered}

Charles University has an official statement on AI use ([ai.cuni.cz/AI-17.html](https://ai.cuni.cz/AI-17.html){.link-no-footnote}) and also a set of informal recommendations for students ([ai.cuni.cz/AI-11.html](https://ai.cuni.cz/AI-11.html){.link-no-footnote}), which essentially allow the use of tools with advanced artificial intelligence.
However, we students have to be transparent about how we use the tools, and we cannot rely solely on AI to do the science nor learning for us.

<!-- Before I
First I thought that it would be easier to just avoid using LLMs, avoid this discussion and avoid the potential debates if I or ChatGPT should get the degree.
Then I realized, I already have half of the software done, with Github Copilot enabled.
With the **Avoid** option greyed out, I might as well try all other models and maybe learn something from it. -->

Github Copilot probably had the largest impact on this work, although it did not write a single word in this PDF.
It is a service developed by Github, owned by Microsoft, based on OpenAI's LLMs and it is a code autocompletion tool.
Copilot automatic completions were very useful to me throughout writing of the codebase, although the code would probably look very similar without the service.

For editing the text of the thesis, I mostly utilized the Mixtral 8x7B Instruct LLM after some initial experimentation with other <del>open-</del>downloadable-weights models.
I also used OpenAI GPT-4 and later 4o for "brainstorming", as a fuzzy search tool, and for helping with troubleshooting of certain issues I am not very familiar with (i.e., Ghostscript and PDF/A).

My favorite Mixtral prompt is “Rewrite the following to formal (academic) English: ”, followed by an unpolished sentence in plain informal English full of typos.
The LLM provides me with a significantly better starting point for subsequent editing, comes up with fancier vocabulary than "then, and, but, nice, ..." and sometimes figures out a better way to formulate the idea.
Finally, I attempted to utilize LLM to perform a review of the entire manuscript sentence-by-sentence, the relevant script is included in the attachment (`thesis/llamacpp-review.mjs`).
<!-- It, however, did not work as well as I would hope; opening the thesis in MS Word yielded better results at a fraction of the computational cost. -->
