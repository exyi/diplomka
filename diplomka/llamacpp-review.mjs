// adapted https://github.com/ggerganov/llama.cpp/blob/master/examples/server/chat.mjs

import * as readline from 'node:readline'
import { stdin, stdout } from 'node:process'
import { readFileSync } from 'node:fs'

const args = process.argv.slice(2);

const no_cached_prompt = args.find(
    (_, index) => args[index - 1] === "--no-cache-prompt"
) ?? "false";

const mode = args.find(
    (_, index) => args[index - 1] === "--mode"
) ?? "doc";
if (!["doc", "chat"].includes(mode)) {
    throw new Error(`Invalid mode: ${mode}, must be doc/chat`);
}

const model = args.find(
    (_, index) => args[index - 1] === "--model"
) ?? 'mistral'
if (!['mistral', 'llama3'].includes(model)) {
    throw new Error(`Invalid model family: ${model}, must be mistral/llama3`)
}

const fullContext = true
const debug = args.some(a => a === "--debug")

const inputFile = args.find((_, index) => args[index - 1] === "--input");

// for cached prompt
let slot_id = -1;

const API_URL = 'http://127.0.0.1:8080'

const rollingContextSize = 10

const cc = {
    reset: "\x1b[0m",
    fgYellow: "\x1b[33m",
    fgCyan: "\x1b[36m",
    fgGreen: "\x1b[32m",
}

const st = model == 'mistral' ? {
        instBegin: '[INST]',
        instEnd: '[/INST]',
        systemRole: '%%%System: ',
        userRole: '%%%User: ',
        assRole: '%%%Assistant: ',

    } :
    model == 'llama3' ? {
        instBegin: '',
        instEnd: '<|eot_id|>',
        systemRole: '<|start_header_id|>system<|end_header_id|>\n\n',
        userRole: '<|start_header_id|>system<|end_header_id|>\n\n',
        assRole: '|start_header_id|>assistant<|end_header_id|>\n\n',
    } : 
    undefined;

const chatInitial = [
    {
        human: "Teh largest city is Europe is Moscow, the capital of Russia.",
        assistant: "'Teh' is a typo, should be 'The'"
    },
    {
        human: "The peptide bonds on DNA are not neccesarily planar.",
        assistant: "'neccesarily' is a typo, should be 'necessarily'; Use 'in DNA' instead of 'on DNA'; DNA does not normally contain peptide bonds, are you sure that is correct?"
    },
    {
        human: "The available software tools are generally mature and provide good quality output.\nHowever, we identified few issues listed below:",
        assistant: ""
    },
]

const chatConstSuffix = [
    {
        human: "![A figure illustrating the phenomenon mentioned in the previous paragraph.](../img/e43f7881-b4f9-4c5d-aecc-c35dd9ce406d.svg)",
        assistant: "Typo in 'illustrating'. Also, consider being more specific in the image caption.",
    }
]

const rollingHistory = []

// const instruction = `${st.instBegin}%%% System: An interaction between a human and an artificial intelligence assistant - a reviewer. The assistant always responds by pointing out mistakes or suspicious statements, if there are any. If there are no major issues, the assistant responds 'No issues.', otherwise writes a concise description of the issues and remedy (for example: 'teh' is a typo, replace with 'the'). The text is in Markdown format, the assistant doesn't check Markdown grammar. The text should be formal (academic) English, the assistant also points out stylistic issues in the text.${st.instEnd}`
const instruction = `${st.instBegin}${st.systemRole}An interaction between a human and an artificial intelligence assistant - a reviewer. The assistant always responds by pointing out mistakes or suspicious statements, if there are any. If there are no major issues, the assistant responds 'No issues.', otherwise writes a concise description of the issues and remedy (for example: 'teh' is a typo, replace with 'the'). The text is in Markdown format, the assistant doesn't check Markdown grammar. The text should be formal (academic) English, the assistant also points out stylistic issues in the text.${st.instEnd}`

const docModeHeader = `
# Reviewed manuscript

This section contains review comments of the above manuscript, where applicable.
The original text is quoted using the ('> xx') markdown syntax, all unquoted text are the comments of the reviewer.
The review may be opinionated, don't take every comment too seriously, but it is hopefully helpful to you.
I think it is better to have "false positive" comments than "false negatives" ;)
I ignored Markdown syntax errors and similar issues, as you can find them easily with a linter.`

const bullshitSentencePrompts = [
    `> Lorem ipsum\n\nProbably remove this.\n\n`,
    `> The largest city is Europe is Moscow, the capital of Russia.\n\nSeems out of place.\n\n`,
]
const fineBullshits = [
    `> TODO zkontrolovat tuto píčovinu`,
    `> <!-- returning 8,783 PDB IDs (as of 16 Oct 2022) -->`,
]

const contextInstruction = ``
const docDropout = 0.7 // 0..1 - probability to include a review in the autoregressive prompt

function format_prompt(question) {
    if (mode == "chat") {
        return `${instruction}\n${
            [...chatInitial, ...rollingHistory, ...chatConstSuffix].map(m =>`%%% Human: ${m.human}\n\n%%% Assistant: ${m.assistant || "No issues."}\n`).join("\n")
        }\n${instruction}\n\n%%% Human: ${question}\n\n%%% Assistant: `
    } else if (mode == "doc") {
        const sentences = [
            ...chatInitial,
            { human: "", assistant: "" },
            ...rollingHistory.map(x => Math.random() < docDropout ? { human: x.human } : x),
            // ...chatConstSuffix
        ];
        const reviewSentences = sentences.map(m => m.assistant ? `${md_quote(m.human)}\n\n${m.assistant}\n\n` : md_quote(m.human));
        for (const b of [...fineBullshits, ...bullshitSentencePrompts]) {
            if (Math.random() > 0.5) {
                reviewSentences.splice(Math.floor(Math.random() * reviewSentences.length), 0, b)
            }
        }
        // console.log(reviewSentences)
        return `
${docModeHeader}

${reviewSentences.join("\n")}
${md_quote(question)}`
    }
}

function md_quote(text) {
    return text.split('\n').map(l => l != "" ? `> ${l}` : "").join('\n')
}

async function tokenize(content) {
    const result = await fetch(`${API_URL}/tokenize`, {
        method: 'POST',
        body: JSON.stringify({ content })
    })

    if (!result.ok) {
        return []
    }

    return await result.json().tokens
}

const n_keep = await tokenize(instruction).length

async function ask_model(fullPrompt) {

    const stop = ["\n%%% Human:", '\n' + st.userRole] // stop completion after generating this
    if (mode == "doc") {
        stop.push("\n>")
        // stop.push("\n\n")
    } else if (mode == "chat") {
    }
    // console.log(cc.fgCyan + fullPrompt + cc.reset, stop)
    const signal = new AbortController()
    const result = await fetch(`${API_URL}/completion`, {
        method: 'POST',
        body: JSON.stringify({
            prompt: fullPrompt,
            temperature: 1.1,
            top_k: 60,
            top_p: 0.95,
            n_keep: n_keep,
            repeat_penalty: 1.1,
            n_predict: 512,
            cache_prompt: no_cached_prompt === "false",
            slot_id: slot_id,
            stop,
            stream: true,
        }),
        signal: signal.signal
    })

    if (!result.ok) {
        process.stdout.write('\n\n ## No response.' + "\n\n")
        return
    }

    let answer = ''

    let prevMessage = ''
    for await (var chunk of result.body) {
        let t = Buffer.from(chunk).toString('utf8')
        if (!t.startsWith('data: ')) {
            t = prevMessage + t
            prevMessage = ''
        }
        if (t.startsWith('data: ')) {
            if (prevMessage != '') {
                process.stdout.write('\n\n ## Error parsing message: ' + t + "\n\n")
            }
            try {
                const message = JSON.parse(t.substring(6))
                slot_id = message.slot_id
                if (message.stop) {
                    if (message.truncated) {
                        // process.stdout.write('\n\n ## Message truncated?? ' + t + "\n\n")
                    }
                    break
                }
                answer += message.content
                if (/^>|^#+\s/.test(message.content) || answer.includes("\n\n")) {
                    // signal.abort()
                    break
                }
                process.stdout.write(message.content)
                prevMessage = ''
            } catch (e) {
                if (debug) console.log(e)
                prevMessage = t
            }
        }
    }

    process.stdout.write('\n')
    return answer
}

async function chat_completion(document, question) {
    while (rollingHistory.length > rollingContextSize) {
        rollingHistory.shift()
    }
    console.log(cc.fgYellow + question + cc.reset)
    const answer = await ask_model(format_prompt(question))

    rollingHistory.push({ human: question, assistant: answer.trim() })
}

async function doc_completion(document, paragraph) {
    const xx = paragraph.split("\n")
    for (let i = 0; i < xx.length; ++i) {
        const line = xx[i]
        console.log(cc.fgYellow + md_quote(line) + cc.reset)
        while (rollingHistory.length > rollingContextSize) {
            rollingHistory.shift()
        }
        let answer = await ask_model(
            `
    ${document.join("\n")}

    ---

    ${format_prompt(line)}\n` + (xx.length - 1 == i ? "\n" : "")
        )

        if (answer.trim().endsWith("\n>")) {
            throw "grr"
        }

        rollingHistory.push({ human: line, assistant: answer.trim() })
    }
    rollingHistory.push({ human: "", assistant: "" })
}

const completion = mode == "doc" ? doc_completion : chat_completion

if (inputFile == null) {
    const rl = readline.createInterface({ input: stdin, output: stdout });

    const readlineQuestion = (rl, query, options) => new Promise((resolve, reject) => {
        rl.question(query, options, resolve)
    });

    async function readlineMultiline(rl, firstQuery, continueQuery, options) {
        const lines = []
        let newLine = null
        while ((newLine = await readlineQuestion(rl, lines.length > 0 ? continueQuery : firstQuery, options))) {
            lines.push(newLine)
        }
        return lines
    }

    while(true) {
        // const question = await readlineQuestion(rl, '> ')
        const question = (await readlineMultiline(rl, '  > ', '... ')).join('\n')
        await completion([], question)
    }
} else {
    const inputData = readFileSync(inputFile, 'utf8').split('\n').map((x) => x.trim())
    const paragraphs = [ [] ]
    for (const line of inputData) {
        if (line == '') {
            if (paragraphs.at(-1).length > 0) {
                paragraphs.push([])
            }
        } else {
            paragraphs[paragraphs.length - 1].push(line)
        }
    }

    for (const paragraph of paragraphs) {
        if (paragraph.length == 0)
            continue;

        process.stdout.write('\n')

        await completion(inputData, paragraph.join('\n'))
        process.stdout.write('\n')
    }
    process.exit(0)
}
