# B. Making Of {.unnumbered}

## The Research Group {.unnumbered}

The thesis is part of a larger team project and therefore has to include a statement detailing which proportion of the work has been done by me.
I largely only describe my own work, leaving out other parts of the project to be published later in a journal.
The notable exceptions are:

* Data filtration methodology of the "Reference Set" was developed by Jiří Černý and the staff of his laboratory.
* The final basepair selection criteria was in a large part prepared by Lada Biedermannová.
* The new basepair parameters were devised in brainstorming sessions with Jiří, Bohdan and Lada and vetted by 
* Valuable feedback provided by all members of the research group has had a noticeable impact on my work, mainly on the new basepair parameters and the web application.

All active research group members were: B. Schneider, J. Černý, L. Biedermannová, H. M. Berman, E. Westhof, C. Zirbel, M. Magnus, and R. Joosten.

## Use of Large Language Models {.unnumbered}

Charles University has an official statement on AI use: [ai.cuni.cz/AI-17.html](https://ai.cuni.cz/AI-17.html){.link-no-footnote} and also a set of informal recommendations for students: [ai.cuni.cz/AI-11.html](https://ai.cuni.cz/AI-11.html){.link-no-footnote}, which essentially allows the use of tools with advanced artificial intelligence.
However, we students have to be transparent about how we used the tools, and we cannot rely solely on AI to do the science or learning for us.

<!-- Before I
First I thought that it would be easier to just avoid using LLMs, avoid this discussion and avoid the potential debates if I or ChatGPT should get the degree.
Then I realized, I already have half of the software done, with Github Copilot enabled.
With the **Avoid** option greyed out, I might as well try all other models and maybe learn something from it. -->

Github Copilot had probably the largest impact on this work, although it didn't write a single word in this PDF.
It is a service developed by Github, owned by Microsoft, and it is a code autocompletion tool, based on OpenAI's LLMs.
I did not utilize the "Copilot chat" feature.

For editing the text of the thesis, I mostly utilized Mixtral 8x7B Instruct run locally using llama.cpp after some initial testing with other <del>open</del> downloadable weights model.
I also used OpenAI GPT-4 and later 4o for "brainstorming", as a fuzzy search tool, and for helping with troubleshooting of certain issues I am not very familiar with (i.e., ghostscript and PDF/A).

My favorite Mixtral prompt is "Rewrite the following to formal (academic) English: ", followed by an unpolished sentence in plain informal English (and full of typos).
The LLM provides me with a significantly better starting point for subsequent editing, and comes up with fancier vocabulary than "then, and, but, nice, ...".
Finally, I attempted to utilize LLM to perform a review of the entire manuscript sentence-by-sentence, the relevant script is included in the attachment (`llamacpp-review.mjs`).
