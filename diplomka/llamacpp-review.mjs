// adapted https://github.com/ggerganov/llama.cpp/blob/master/examples/server/chat.mjs

import * as readline from 'node:readline'
import { stdin, stdout } from 'node:process'
import { readFileSync } from 'node:fs'

const args = process.argv.slice(2);

const no_cached_prompt = args.find(
    (_, index) => args[index - 1] === "--no-cache-prompt"
) ?? "false";

const inputFile = args.find((_, index) => args[index - 1] === "--input");

// for cached prompt
let slot_id = -1;

const API_URL = 'http://127.0.0.1:8080'

const chatMaxSize = 10

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
        assistant: "No issues."
    },
]

const chatConstSuffix = [
    {
        human: "![A figure illustrating the phenomenon mentioned in the previous paragraph.](../img/e43f7881-b4f9-4c5d-aecc-c35dd9ce406d.svg)",
        assistant: "Typo in 'illustrating'. Also, consider being more specific in the image caption.",
    }
]

const chat = []

const instruction = `[INST]%%% System: An interaction between a human and an artificial intelligence assistant - a reviewer. The assistant always responds by pointing out mistakes or suspicious statements, if there are any. If there are no major issues, the assistant responds 'No issues.', otherwise writes a concise description of the issues and remedy (for example: 'teh' is a typo, replace with 'the'). The text is in Markdown format, the assistant doesn't check Markdown grammar. The text should be formal (academic) English, the assistant also points out stylistic issues in the text.[/INST]`

function format_prompt(question) {
    return `${instruction}\n${
        [...chatInitial, ...chat, ...chatConstSuffix].map(m =>`%%% Human: ${m.human}\n\n%%% Assistant: ${m.assistant}\n`).join("\n")
    }\n${instruction}\n\n%%% Human: ${question}\n\n%%% Assistant: `
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

async function chat_completion(question) {
    while (chat.length > chatMaxSize) {
        chat.shift()
    }
    const result = await fetch(`${API_URL}/completion`, {
        method: 'POST',
        body: JSON.stringify({
            prompt: format_prompt(question),
            temperature: 0.2,
            top_k: 60,
            top_p: 0.95,
            n_keep: n_keep,
            repeat_penalty: 1.1,
            n_predict: 386,
            cache_prompt: no_cached_prompt === "false",
            slot_id: slot_id,
            stop: ["\n%%% Human:"], // stop completion after generating this
            stream: true,
        })
    })

    if (!result.ok) {
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
                answer += message.content
                process.stdout.write(message.content)
                if (message.stop) {
                    if (message.truncated) {
                        chat.shift()
                    }
                    break
                }
                prevMessage = ''
            } catch (e) {
                prevMessage = t
            }
        }
    }

    process.stdout.write('\n')
    chat.push({ human: question, assistant: answer.trimStart() })
}

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
        await chat_completion(question)
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

        for (const line of paragraph) {
            process.stdout.write('> ' + line + '\n')
        }
        process.stdout.write('\n')

        await chat_completion(paragraph.join('\n'))
        process.stdout.write('\n')
    }
    process.exit(0)
}
