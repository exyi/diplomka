
import type * as arrow from 'apache-arrow'
import type { VariableModel } from './dbModels'
import type { TypedArray } from 'd3'
import { Lazy } from './lazy'
import _ from 'lodash'

// export type JsFlatTypeArray = Float64Array | Int32Array | Uint8Array | Uint16Array | Uint32Array | Int8Array | Int16Array | Int32Array
export type FloatArray = Float32Array | Float64Array
export type NumArray = FloatArray | Int32Array | Uint32Array | Uint16Array | Int16Array | Uint8Array | Int8Array
export type SmallIntArray = Int32Array | Uint32Array | Uint16Array | Int16Array | Uint8Array | Int8Array

function andBitmap(destination: Uint8Array, source: Uint8Array, destOffset) {
    if (destOffset % 8 == 0) {
        for (let i = 0; i < source.length; i++) {
            destination[(destOffset >> 3) + i] &= source[i]
        }
    } else {
        console.warn("andBitmap: unaligned at", destOffset)
        // throw new Error("Not implemented")
        const len = Math.min(destination.length * 8 - destOffset, source.length * 8)
        for (let i = 0; i < len; i++) {
            const bit = ((source[i >> 3] >> (i & 7)) & 1)
            destination[(destOffset + i) >> 3] &= ~((1 - bit) << ((destOffset + i) & 7))
        }
    }
}
function popcountBitmap(bitmap: Uint8Array, size: number) {
    let count = 0;
    for (let i = 0; i < size; i++) {
        if ((bitmap[i >> 3] & (1 << (i & 7))) != 0) {
            count++
        }
    }
    return count
}

type DataList<T extends arrow.DataType<arrow.Type, any>> = readonly arrow.Data<T>[]

function makeFilterBitmap(size: number, filters: readonly DataList<arrow.Bool>[], nnfilters: readonly DataList<any>[]): Uint8Array {
    const bitmap = new Uint8Array(Math.ceil(size / 8))
    bitmap.fill(0xff)

    for (const filter of filters) {
        for (let i = 0, offset = 0; i < filter.length; offset += filter[i].length, i++) {
            const d = filter[i]
            if (d.nullable) {
                andBitmap(bitmap, d.nullBitmap, offset)
            }
            andBitmap(bitmap, d.values, offset)
        }
    }

    for (const nnfilter of nnfilters) {
        for (let i = 0, offset = 0; i < nnfilter.length; offset += nnfilter[i].length, i++) {
            const d = nnfilter[i]
            if (d.nullable) {
                andBitmap(bitmap, d.nullBitmap, offset)
            }
        }
    }
    return bitmap
}

export function filterData(ArrayType: typeof Float64Array | typeof Float32Array, data: DataList<any>, filters: readonly DataList<arrow.Bool>[], nnfilters: readonly DataList<any>[]): [Float64Array | Float32Array, number, Uint8Array] {
    const totalCount = data.reduce((acc, cur) => acc + cur.length, 0)
    if (!nnfilters.includes(data) && data.some(d => d.nullable)) {
        nnfilters = [...nnfilters, data]
    }
    const bitmap = makeFilterBitmap(totalCount, filters, nnfilters)

    const result = new ArrayType(popcountBitmap(bitmap, totalCount))
    for (let di = 0, ri = 0, si = 0; di < data.length; di++) {
        const d = data[di]
        for (let i = 0; i < d.length; i++, si++) {
            if ((bitmap[si >> 3] & (1 << (si & 7))) != 0) {
                result[ri++] = Number(d.values[i])
            }
        }
    }
    return [result, totalCount, bitmap]
}

export function filterTransposeBools(dataLists: DataList<arrow.Bool>[], filters: readonly DataList<arrow.Bool>[], nnfilters: readonly DataList<any>[], nullValue: boolean): [Uint32Array, number, Uint8Array] {
    if (dataLists.length > 32) {
        throw new Error("Too many boolean columns")
    }
    const totalCount = dataLists[0].reduce((acc, cur) => acc + cur.length, 0)
    for (const data of dataLists) {
        if (data.reduce((acc, cur) => acc + cur.length, 0) != totalCount) {
            throw new Error("Data length mismatch")
        }
    }
    const filterBitmap = makeFilterBitmap(totalCount, filters, nnfilters)

    const result = new Uint32Array(popcountBitmap(filterBitmap, totalCount))
    result.fill(0)

    for (let dli = 0; dli < dataLists.length; dli++) {
        const data = dataLists[dli]
        for (let di = 0, ri = 0, si = 0; di < data.length; di++) {
            const d = data[di]
            for (let i = 0; i < d.length; i++, si++) {
                if ((filterBitmap[si >> 3] & (1 << (si & 7))) != 0) {
                    const isNull = d.nullable && (d.nullBitmap[i >> 3] & (1 << (i & 7))) > 0
                    const bit = isNull ? Number(nullValue) : ((d.values[i >> 3] >> (i & 7)) & 1)
                    result[ri++] |= bit << dli
                }
            }
        }
    }
    return [result, totalCount, filterBitmap]
}

export function getIndexIndex(bitmap: Uint8Array, count: number): Int32Array {
    const index = new Int32Array(count)
    for (let i = 0, j = 0; i < count; i++) {
        if ((bitmap[i >> 3] & (1 << (i & 7))) != 0) {
            index[i] = j++
        } else {
            index[i] = -1
        }
    }
    return index
}

export function getColumnHelper(table: arrow.Table, variable: VariableModel, shareFilters: VariableModel[] = [variable]) {
    const x = tryGetColumnHelper(table, variable, shareFilters)
    if (x == null) {
        throw new Error(`Column ${variable.column} not found!`)
    }
    return x
}

export function tryGetColumnHelper(table: arrow.Table, variable: VariableModel, shareFilters: VariableModel[] = [variable]) {
    const column = table.getChild(variable.column)
    if (column == null) {
        return null
    }
    if (!shareFilters.includes(variable)) {
        throw new Error("shareFilters must include itself")
    }

    const filters = shareFilters.flatMap(v => v.filterId ? [table.getChild(v.filterId).data] : [])
    const nnfilters = shareFilters.filter(v => v != variable).map(v => table.getChild(v.column).data)
    const filter = variable.filterId ? table.getChild(variable.filterId) : null
    const filtered = new Lazy(() => filterData(Float32Array, column.data, filters, nnfilters))
    return {
        arrowColumn: column,
        arrowFilter: filter,

        get data() {
            return filtered.value[0] as Float32Array
        },
        get totalRowCount() {
            if (filtered.isComputed) {
                if (filtered.value[1] != column.length) {
                    throw new Error("WTF")
                }
            }
            return column.length
        },
        get filterOutBitmap() {
            return filtered.value[2]
        },
        get indexIndex() {
            return getIndexIndex(filtered.value[2], column.length)
        }
    }
}

export function tryGetBoolColumn(table: arrow.Table, variables: Omit<VariableModel, "label">[], shareFilters: Omit<VariableModel, "label">[] = [], nullValue: boolean | null = null) {
    const filters = [...shareFilters, ...variables].flatMap(v => v.filterId ? [table.getChild(v.filterId)?.data] : [])
    const varData = variables.map(v => table.getChild(v.column)?.data)
    const nnfilters = shareFilters.map(v => table.getChild(v.column)?.data)

    if (varData.some(v => v == null) || nnfilters.some(v => v == null) || filters.some(v => v == null)) {
        return null
    }

    const processed = new Lazy(() => filterTransposeBools(varData, filters, nnfilters, nullValue ?? false))
    return {
        get data() {
            return processed.value[0] as Uint32Array
        },
        get totalRowCount() {
            return processed.value[1]
        },
        get filterOutBitmap() {
            return processed.value[2]
        },
    }
}

export type ColumnHelper = ReturnType<typeof getColumnHelper>

export function concatArrays<T extends TypedArray>(arrays: readonly T[]): T {
    const length = arrays.reduce((acc, cur) => acc + cur.length, 0)
    const ctor = arrays[0].constructor as any
    const result:T = new ctor(length)
    for (let i = 0, offset = 0; i < arrays.length; i++, offset += arrays[i].length) {
        result.set(arrays[i], offset)
    }
    return result
}

function clamp(x: number, min: number, max: number) {
    return Math.min(max, Math.max(min, x))
}


export function binArrays(data: FloatArray, min, max, binCount) {
    const result = new Uint32Array(binCount).fill(0)
    const binSize = (max - min) / binCount
    for (let i = 0; i < data.length; i++) {
        const bin = Math.min(binCount - 1, Math.floor((data[i] - min) / binSize))
        result[clamp(bin, 0, result.length - 1)]++
    }
    return result
}

export function binArray2D(x: FloatArray, y: FloatArray, minX, maxX, minY, maxY, binCountX, binCountY) {
    const result = new Uint32Array(binCountX * binCountY).fill(0)
    const binSizeX = (maxX - minX) / binCountX
    const binSizeY = (maxY - minY) / binCountY
    for (let i = 0; i < x.length; i++) {
        const binX = clamp(Math.round((x[i] - minX) / binSizeX), 0, binCountX - 1)
        const binY = clamp(Math.round((y[i] - minY) / binSizeY), 0, binCountY - 1)
        result[binX + binY * binCountX]++
    }
    return result
}

export function binArrayCateg(x: FloatArray, y: SmallIntArray, min, max, binCount) {
    const categos = new Map<number, Uint32Array>()
    const binSize = (max - min) / binCount
    for (let i = 0; i < x.length; i++) {
        let a = categos.get(y[i])
        if (a == null) {
            categos.set(y[i], a = new Uint32Array(binCount).fill(0))
        }
        const bin = Math.min(binCount - 1, Math.floor((x[i] - min) / binSize))
        a[clamp(bin, 0, a.length - 1)]++
    }
    return categos
}

export function arange(start: number, stop: number, step = 1): Int32Array {
    const result = new Int32Array(Math.ceil((stop - start) / step))
    for (let i = 0; i < result.length; i++) {
        result[i] = start + i * step
    }
    return result
}

export function stddev(arr: Float64Array | Float32Array) {
    if (arr.length <= 1) return 1;

    let mean = 0
    for (let i = 0; i < arr.length; i++) {
        mean += arr[i]
    }
    mean /= arr.length
    let dev = 0
    for (let i = 0; i < arr.length; i++) {
        dev += (arr[i] - mean) ** 2
    }
    dev /= arr.length - 1
    return Math.sqrt(dev)
}
