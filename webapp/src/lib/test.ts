async function streamingParse(url) {
    const respone = await fetch(url)
    const reader = respone.body.getReader({mode: 'byob'})

    let buffer = new Uint8Array(4096)
    let remaining = 0

    for (;;) {
        const r = await reader.read(buffer.subarray(remaining))
        if (r.done) {
            break
        }
        const readLength = r.value.length + remaining
        let index = 0, nextIndex = 0;
        while ((nextIndex = buffer.indexOf('\n'.charCodeAt(0), index)) >= 0) {
            const line = buffer.slice(index, nextIndex)
            console.log(new TextDecoder().decode(line))
            index = nextIndex + 1
        }
        remaining = readLength - index
        if (remaining == buffer.length) {
            console.log('Line too long')
            const newBuffer = new Uint8Array(buffer.length * 2)
            newBuffer.set(buffer)
            buffer = newBuffer
        }
        buffer.copyWithin(0, index, readLength)
    }
}
