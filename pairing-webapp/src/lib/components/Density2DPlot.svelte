<script lang="ts">
    import type { KDE2DSettingsModel } from "$lib/dbModels";
    import * as d3 from "d3";
    import type * as arrow from 'apache-arrow'
	import * as a from "$lib/arrowUtils";
	import { onMount } from "svelte";
    export let settings: KDE2DSettingsModel | null = null
    export let data: arrow.Table | null = null

    let svg: SVGGElement, canvas: HTMLCanvasElement
    const widthPx = 640
    const heightPx = 480
    const marginPx = {top: 50, right: 50, bottom: 40, left: 50}
    const colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']

    function getColor(column: string) {
        let m
        if (m = /^hb_(\d+)/.exec(column)) {
            return colors[parseInt(m[1]) % colors.length]
        } else {
            return null
        }
    }

    // function generateHistogramImg(data: Uint32Array, color: string, width: number, height: number) {
    //     if (data.length != width * height) {
    //         throw 'ne'
    //     }

    //     const canvas = document.createElement('canvas')
    //     canvas.width = width
    //     canvas.height = height
    //     const ctx = canvas.getContext('2d')!
    //     ctx.fillStyle = "#ffffff";
    //     ctx.fillRect(0, 0, width, height);
    //     const imageData = ctx.createImageData(width, height)
    //     const colorBuffer = new Uint32Array(imageData.data.buffer)

    //     const colorRgb = d3.color(color).rgb()
    //     const colorInt = (colorRgb.r << 16) | (colorRgb.g << 8) | colorRgb.b

    //     const max = data.reduce((a, b) => Math.max(a, b), 0)
    //     const ascale = 255 / max

    //     for (let i = 0; i < data.length; i++) {
    //         const c = data[i]
    //         const a = Math.min(255, c * ascale) | 0
    //         colorBuffer[i] = (a << 24) | colorInt
    //     }

    //     ctx.putImageData(imageData, 0, 0)
    //     return canvas.toDataURL("image/png")
    // }

    function drawScatter(xs: Float64Array | Float32Array, ys: Float64Array | Float32Array, xRange: number[], yRange: number[], colorLabels: Uint32Array | null, colorMapping: Map<number, string>, defaultColor: string) {
        canvas.width = canvas.parentElement.clientWidth
        canvas.height = canvas.parentElement.clientHeight
        const ctx = canvas.getContext('2d')!
        ctx.fillStyle = "rgba(0, 0, 0, 0)"
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        const [xMin, xMax] = xRange,
              [yMin, yMax] = yRange

        const crossSize = xs.length < 300 ? 5 :
            xs.length < 1000 ? 3 :
            xs.length < 5000 ? 2 :
            1

        const colorMap2 = new Map<number, string>()
        function procColor(color: string) {
            const c = d3.color(color).rgb()
            return `rgba(${c.r}, ${c.g}, ${c.b}, 0.5)`
        }
        let lastColorId = -1
        let lastColor = defaultColor
        function switchColor(i: number) {
            if (colorLabels == null) return

            const colorId = colorLabels[i]
            if (lastColorId == colorId) return
            let color = colorMap2.get(colorId)
            if (color == null) {
                colorMap2.set(colorId, color = procColor(colorMapping.get(colorId) ?? defaultColor))
            }
            lastColorId = colorId
            if (color != lastColor) {
                ctx.stroke()
                ctx.strokeStyle = ctx.fillStyle = color
                ctx.beginPath()
                lastColor = color
            }
        }

        ctx.strokeStyle = ctx.fillStyle = procColor(defaultColor)
        ctx.lineWidth = xs.length < 300 ? 2 : 1
        ctx.beginPath()
        for (let i = 0; i < xs.length; i++) {
            switchColor(i)
            const x = (xs[i] - xMin) / (xMax - xMin) * canvas.width
            const y = canvas.height - (ys[i] - yMin) / (yMax - yMin) * canvas.height
            if (crossSize <= 1) {
                ctx.fillRect(x, y, 1, 1)
            } else {
                ctx.moveTo(x - crossSize, y - crossSize)
                ctx.lineTo(x + crossSize, y + crossSize)
                ctx.moveTo(x - crossSize, y + crossSize)
                ctx.lineTo(x + crossSize, y - crossSize)
            }
        }
        ctx.stroke()
    }

    function rollingMeans(x: Float64Array, y: Float64Array, minY, maxY, step) {
        const means = new Float64Array((maxY - minY) / step + 1)
        const counts = new Uint32Array((maxY - minY) / step + 1)

        for (let i = 0; i < x.length; i++) {
            const yIdx = Math.floor((y[i] - minY) / step)
            means[yIdx] += x[i]
            means[yIdx + 1] += x[i]
            counts[yIdx]++
            counts[yIdx + 1]++
        }
        for (let i = 0; i < means.length; i++) {
            means[i] /= counts[i]
        }
        return means
    }

    function miniHistogram(node: d3.Selection<SVGGElement, unknown, null, undefined>, bins: Uint32Array[], height, width, colors: string[]) {
        const count = bins[0].length
        const transposed: Uint32Array[] = []
        for (let i = 0; i < bins[0].length; i++) {
            const a = new Uint32Array(bins.length)
            for (let j = 0; j < bins.length; j++) {
                a[j] = bins[j][i]
            }
            transposed.push(a)
        }
        const maxHeight = transposed.map(a => a.reduce((a, b) => a+b, 0)).reduce((a, b) => Math.max(a, b), 0)
        const offsets = new Uint32Array(count).fill(0)
        for (let i = 0; i < bins.length; i++) {
            node.append("g")
                .selectAll("rect")
                .data(bins[i])
                .enter()
                .append("rect")
                .attr("data-color-index", i)
                .attr("x", (d, i) => i * width / count)
                .attr("y", (d, i) => height - (d+offsets[i]) / maxHeight * height)
                .attr("width", function(d) { return width / count })
                .attr("height", function(d) { return d / maxHeight * height })
                .style("fill", d3.color(colors[i]).brighter(0.5).formatHex())
                .style("stroke", d3.color(colors[i]).darker(0.5).formatHex())
                .style("fill-opacity", 0.6)
            
            for (let j = 0; j < bins[i].length; j++) {
                offsets[j] += bins[i][j]
            }
        }
    }

    function miniHistogramCalc(node: d3.Selection<SVGGElement, unknown, null, undefined>, height, width, data: a.FloatArray, min, max, reverse, defaultColor: string, categos: Uint32Array | null, colorMapping: Map<number, string>, colorOrder: number[] | null) {
        if (categos == null) {
            const bins = a.binArrays(data, min, max, 60)
            miniHistogram(node, [bins], height, width, [defaultColor])
            return
        }

        const bins = a.binArrayCateg(data, categos, min, max, 60)
        const keys = colorOrder?.filter(k => colorMapping.has(k)) || Array.from(colorMapping.keys()).toSorted((a, b) => a - b)
        miniHistogram(node, keys.map(k => bins.get(k)).map(x => reverse ? x.toReversed() : x), height, width, keys.map(k => colorMapping.get(k) ?? defaultColor))
    }

    function rerender() {
        const startTime = performance.now()
        if (!settings) return
        if (settings.variables.length < 2) {
            console.error("density plot requires at least two variables")
            return
        }
        if (settings.variables.length > 2) {
            console.warn("density plot only currently renders only the first two variables")
        }
        if (settings.variables.some(v => !v.column)) {
            return
        }

        const variables = settings.variables.slice(0, 2)
        
        const comparisonMode = data.getChild("comparison_in_baseline") != null && data.getChild("comparison_in_current") != null
        const colorLabels =
            comparisonMode ? a.tryGetBoolColumn(data, [ { column: "comparison_in_baseline" }, { column: "comparison_in_current" } ], variables)
                           : null
        const colorMapping = new Map<number, string>()
        if (comparisonMode) {
            colorMapping.set(0b11, "#0000ff")
            colorMapping.set(0b01, "#ff0000")
            colorMapping.set(0b10, "#008700")
        }

        const color = getColor(variables[0].column) ?? getColor(variables[1].column) ?? '#0000ff'
        const colorY = getColor(variables[1].column) ?? color

        const columns = variables.map(v => a.tryGetColumnHelper(data, v, variables)) as [a.ColumnHelper, a.ColumnHelper]
        if (columns.some(c => c == null)) {
            console.warn("2D plot: missing columns in the table")
            return
        }
        const x = columns[0].data,
              y = columns[1].data
        if (x.length != y.length) {
            console.error("2D plot requires variables to have the same length", variables, columns)
            return
        }
        if (x.length <= 5) {
            return
        }

        const xAxis = d3.scaleLinear()
            .domain([
                x.reduce((a, b) => Math.min(a, b), 1e50),
                x.reduce((a, b) => Math.max(a, b), -1e50)
            ])
            .nice(40)
            .range([marginPx.left, widthPx - marginPx.right])
        const yAxis = d3.scaleLinear()
            .domain([
                y.reduce((a, b) => Math.min(a, b), 1e50),
                y.reduce((a, b) => Math.max(a, b), -1e50)
            ])
            .nice(40)
            .range([heightPx - marginPx.bottom, marginPx.top])

        const xScaled = x.map(x => xAxis(x))
        const yScaled = y.map(y => yAxis(y))
        const xStd = a.stddev(x)
        const yStd = a.stddev(y)

        const bandwidth = 0.5 * ((x.length)**(-1./(2+4))) * Math.max(xStd * Math.abs(xAxis.range()[1] - xAxis.range()[0]), yStd * Math.abs(yAxis.range()[1] - yAxis.range()[0]))

        const bins2d = a.binArray2D(x, y, xAxis.domain()[0], xAxis.domain()[1], yAxis.domain()[0], yAxis.domain()[1], 60, 60)

        const binsX = a.binArrays(x, xAxis.domain()[0], xAxis.domain()[1], 60)
        const binsY = a.binArrays(y, yAxis.domain()[0], yAxis.domain()[1], 60)
        // const bitmap = generateHistogramImg(bins2d, color, 60, 60)

        // const means = rollingMeans(x, y, yAxis.domain()[0], yAxis.domain()[1], (yAxis.domain()[1] - yAxis.domain()[0]) / 10)

        // const contours = d3.contourDensity<number>()
        //     .x(i => xScaled[i])
        //     .y(i => yScaled[i])
        //     .size([widthPx - marginPx.left - marginPx.right, heightPx - marginPx.top - marginPx.bottom])
        //     .bandwidth(bandwidth)
        //     .cellSize(2)
        //     .thresholds(20)
        //     (a.arange(0, xScaled.length) as any)

        drawScatter(x, y, xAxis.domain(), yAxis.domain(), colorLabels?.data, colorMapping, color)

        const node = d3.select(svg)
        node.selectAll("*").remove()
        node.append("g")
            .attr("transform", `translate(0,${heightPx - marginPx.bottom})`)
            .call(d3.axisBottom(xAxis))
        node.append("g")
            .attr("transform", `translate(${marginPx.left},0)`)
            .call(d3.axisLeft(yAxis))
            .append("text")
            .attr("x", "2")
            .attr("y", "18")
            .attr("dy", ".71em")
            // .text(settings.variables[1].label || settings.variables[1].column)

        node.append("text")
            .attr("class", "")
            .attr("text-anchor", "middle")
            .attr("x", widthPx / 2)
            .attr("y", heightPx - 12)
            .attr("dy", ".71em")
            .text(settings.variables[0].label || settings.variables[0].column)
        node.append("text")
            .attr("class", "")
            .attr("text-anchor", "middle")
            .attr("y", 6)
            .attr("x", (-heightPx) / 2)
            .attr("dy", ".71em")
            .attr("transform", "rotate(-90)")
            .text(settings.variables[1].label || settings.variables[1].column)

        // x-axis histogram
        miniHistogramCalc(
            node.append("g")
                .attr("transform", `translate(${marginPx.left},0)`),
            marginPx.top,
            widthPx - marginPx.left - marginPx.right,
            x,
            xAxis.domain()[0], xAxis.domain()[1],
            false,
            color,
            colorLabels?.data,
            colorMapping,
            comparisonMode ? [ 0b01, 0b11, 0b10 ] : null
        )
        miniHistogramCalc(
            node.append("g")
                .attr("transform", `rotate(90, ${widthPx}, ${marginPx.top}) translate(${widthPx},${marginPx.top})`),
            marginPx.right,
            heightPx - marginPx.top - marginPx.bottom,
            y,
            yAxis.domain()[0], yAxis.domain()[1],
            true,
            colorY,
            colorLabels?.data,
            colorMapping,
            comparisonMode ? [ 0b01, 0b11, 0b10 ] : null
        )

        // node.append("g")
        //     .attr("fill", "none")
        //     .attr("stroke", d3.color(color).darker(1).hex())
        //     .attr("stroke-linejoin", "round")
        //     .selectAll()
        //     .data(contours)
        //     .join("path")
        //     .attr("stroke-width", (d, i) => i % 5 ? 0.25 : 1)
        //     .attr("d", d3.geoPath(d3.geoTransform({
        //         // point(x, y) {
        //         //     this.stream.point(xAxis(x * xStd), yAxis(y * yStd))
        //         // }
        //     })));
        // node.append("g")
        //     .append("path")
        //     .attr("stroke", "red")
        //     .attr("stroke-width", 3)
        //     .attr("fill", "none")
        //     .attr("d", d3.line()(Array.from(means).flatMap((x, i) => {
        //         if (Number.isNaN(y)) return []
        //         // if (i == 0) return [x, yAxis.range()[0]]
        //         return [[xAxis(x), (yAxis.range()[0] * (rollingMeans.length - 1 - i) + yAxis.range()[1] * (i)) / (rollingMeans.length - 1)]]
        //     })))
        // node.append("image")
        //     .attr("x", marginPx.left)
        //     .attr("y", -(heightPx - marginPx.bottom))
        //     .attr("width", widthPx - marginPx.left - marginPx.right)
        //     .attr("height", heightPx - marginPx.top - marginPx.bottom)
        //     .attr("href", bitmap)
        //     .attr("style", "image-rendering: pixelated; transform: scaleY(-1);")
        console.log(`KDE plot rendered in ${performance.now() - startTime}ms`)
    }

    onMount(async () => {
        await 0
        try {
            rerender()
        } catch (e) {
            console.warn(e)
        }
    })

    $: {
        if (settings && data && svg) {
            Promise.resolve().then(rerender).catch(console.warn)
        }
    }
</script>

<style>
    .density-plot {
        position: relative;
        /* width: 100%; */
    }
    svg {
        width: 100%;
        height: 100%;
    }
</style>

<div class="density-plot">
    <svg viewBox="0 0 {widthPx} {heightPx}">
        <g bind:this={svg}></g>
        <!-- <g transform="translate({marginPx.left},{marginPx.top})"> -->
            <!-- </g> -->
        <foreignObject x={marginPx.left} y={marginPx.top} width={widthPx - marginPx.left - marginPx.right} height={heightPx - marginPx.top - marginPx.bottom}>
            <div style="width:100%; height:100%">
                <canvas bind:this={canvas}></canvas>
            </div>
        </foreignObject>

        <!-- <g>
        {#each settings?.variables ?? [] as v, i}
        {#if v.label != null}
            <circle cx={widthPx - marginPx.right - 100} cy={marginPx.top + 30 * i + 6} r="6" style="fill: {d3.color(colors[i]).brighter(0.5).hex() ?? '#000000'}; fill-opacity: 0.6; stroke: {d3.color(colors[i]).darker(0.5).hex()}"/>
            <text x={widthPx - marginPx.right - 90} y={marginPx.top + 30 * i + 12} text-anchor="start" alignment-baseline="middle">
                <title>{v.column} {#if v.filterSql}WHERE {v.filterSql}{/if}</title>
                {v.label}
            </text>
        {/if}
        {/each}
        </g> -->
    </svg>
</div>
