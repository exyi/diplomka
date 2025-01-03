<script lang="ts">
	import { filterToSqlCondition, getColumnLabel, getDataSourceTable, hideColumn, longNucleotideNames, type NucleotideFilterModel, type NumRange } from "$lib/dbModels";
	import metadata from "$lib/metadata";
	import { convertQueryResults, type NucleotideId, type PairId, type PairingInfo } from "$lib/pairing";
    import * as filterLoader from '$lib/predefinedFilterLoader'
	import _ from "lodash";
    import * as dbInstance from '$lib/dbInstance'
	import { ensureViews } from "$lib/dataSourceTables";
	import config from "$lib/config";
	import MolStarMaybe from "./MolStarMaybe.svelte";

    export let imageUrl: string | undefined
    export let rotImageUrl: string | undefined
    export let videoUrl: string | undefined
    export let pair: PairingInfo
    export let pairType: string
    export let filter: NucleotideFilterModel | undefined = undefined
    export let filterBaseline: NucleotideFilterModel | undefined = undefined
    export let requeryDB: boolean = true
    let realFilter: NucleotideFilterModel | undefined = undefined
    let filterLabel: string | undefined = undefined
    let pairDB: PairingInfo | undefined = undefined
    let detailedTable: boolean = false // include likelihood, mode dev
    $: { filter; updateRealFilter() }

    async function updateRealFilter() {
        realFilter = filter
        filterLabel = undefined
        if (["allcontacts-boundaries-f", "allcontacts-boundaries"].includes(filter.datasource)) {
            await filterLoader.defaultFilterLimits.value.then(v => {
                filterLabel = v.id
                realFilter = filterLoader.addHBondLengthLimits(pairType, 0.01, filterLoader.toNtFilter(v.limits, 0, pairType, filter))
                console.log("realFilter", realFilter)
            })
        }
    }

    $: { imageUrl; pair; pairType; filter; requeryDB; updatePairFromDb() }

    async function updatePairFromDb() {
        if (!requeryDB || filter == null || pair == null || pairType == null) {
            pairDB = pair;
            return
        }
        let dataSourceTable = pair.originalRow.comparison_in_baseline === true && filterBaseline ? getDataSourceTable(filterBaseline) : getDataSourceTable(filter)
        if (dataSourceTable == 'selectedpair_allcontacts_boundaries_f') {
            // avoid issues when it isn't found because the filter does not match anymore...
            dataSourceTable = 'selectedpair_allcontacts_f'
        } else if (dataSourceTable == 'selectedpair_allcontacts_boundaries') {
            dataSourceTable = 'selectedpair_allcontacts'
        }
        const query = `
            SELECT * FROM ${dataSourceTable}
            WHERE lower(pdbid) = '${pair.id.nt1.pdbid.toLowerCase()}'
              AND model = ${pair.id.nt1.model}
              AND chain1 = '${pair.id.nt1.chain}'
              AND nr1 = ${pair.id.nt1.resnum}
              AND coalesce(alt1,'') = '${pair.id.nt1.altloc ?? ''}'
              AND coalesce(ins1, '') = '${pair.id.nt1.inscode ?? ''}'
              AND chain2 = '${pair.id.nt2.chain}'
              AND nr2 = ${pair.id.nt2.resnum}
              AND coalesce(alt2,'') = '${pair.id.nt2.altloc ?? ''}'
              AND coalesce(ins2, '') = '${pair.id.nt2.inscode ?? ''}'
              AND coalesce(symmetry_operation1,'') = '${pair.id.nt1.symop ?? ''}'
              AND coalesce(symmetry_operation2,'') = '${pair.id.nt2.symop ?? ''}'
            LIMIT 1`

        const db = dbInstance.getConnectionSync()
        await ensureViews(db, new AbortController().signal, [dataSourceTable], pairType)
        const table = await db.query(query)
        pairDB = [...convertQueryResults(table, pairType, 1)][0]
        if (!pairDB) {
            console.error("Pair not found in database", pair.id)
            pairDB = pair
        }
    }

    const allowMolstar = false

    let avifError = false,
        imgError = false,
        avifRotError = false,
        videoError = false,
        rotError = false,
        molStar = false
    $: { videoUrl; videoError = false; rotError = false; molStar = false }
    $: { rotImageUrl; avifRotError = false; rotError = false; molStar = false }
    $: { imageUrl; avifError = false; imgError = false; molStar = false }

    function getRange(f: NucleotideFilterModel, column: string) : NumRange | undefined {
        if (!f) return undefined
        if (f.other_column_range && column in f.other_column_range) {
            return f.other_column_range[column]
        }
        if (column in f) {
            return f[column]
        }

        if (/^C1_C1_(yaw|pitch|roll)\d*/.test(column)) {
            return f[column.slice("C1_C1_".length)]
        }
    }

    function isOutOfRange(value: number | null | undefined, range: NumRange | null | undefined) {
        if (value == null)
            return null

        if (range?.min == null && range?.max == null)
            return null

        if (range.min != null && range.max != null && range.min > range.max) {
            // range in modular arithmetic (angles -180..180)
            return value > range.min || value < range.max
        }

        if (range?.min != null && value < range.min)
            return false
        if (range?.max != null && value > range.max)
            return false
        return true
    }

    function getRangeValueTitle(value: number | null | undefined, range: NumRange | null | undefined) {
        if (value == null || range == null)
            return ""

        if (range?.min == null && range?.max == null)
            return null

        const inRange = isOutOfRange(value, range)

        if (inRange === true) {
            return `Value ${value?.toFixed(2)} is within range ${range.min?.toFixed(2) ?? '-∞'}..${range.max?.toFixed(2) ?? '∞'}`
        }
        if (inRange === false) {
            return `Value ${value?.toFixed(2)} is outside range ${range.min?.toFixed(2) ?? '-∞'}..${range.max?.toFixed(2) ?? '∞'}`
        }
        return "???"
    }

    function resSele(r: NucleotideId) {
        let nt = String(r.resnum).replace("-", "\\-")
        if (r.inscode) {
            nt += String(r.inscode)
        }

        const alt = r.altloc ? ` and alt ${r.altloc}` : ""
        return `resi ${nt}${alt}`
    }
    
    function generatePymolScript(s: PairId): string[] {
        const script = []
        if (s.nt1.symop || s.nt2.symop) {
            script.push(`set assembly = 1`)
        }
        script.push(`fetch ${s.nt1?.pdbid}`)
        if (s.nt1.model > 1) {
            script.push(`set state, ${s.nt1.model}`)
        }
        const pairSelection =
            String(s.nt1?.chain) != String(s.nt2?.chain) ?
                `${s.nt1?.pdbid} and (chain ${s.nt1.chain} and ${resSele(s.nt1)} or chain ${s.nt2.chain} and ${resSele(s.nt2)})` :
            s.nt1?.altloc || s.nt1?.inscode || s.nt2?.altloc || s.nt2?.inscode || s.nt1.resnum < 0 || s.nt2.resnum < 0 ?
                `${s.nt1?.pdbid} and chain ${s.nt1?.chain} and (${resSele(s.nt1)} or ${resSele(s.nt2)})` :
                `${s.nt1?.pdbid} and chain ${s.nt1?.chain} and resi ${s.nt1.resnum}+${s.nt2.resnum}`
        script.push(`select pair, ${pairSelection}`)
        script.push(`show sticks, %pair`)
        script.push(`orient %pair`)
        script.push(`hide everything, not %pair`)

        return script
    }

    const columnBlacklist = [ "pdbid", "model", "chain1", "chain2", "res1", "res2", "nr1", "nr2", "alt1", "alt2", "ins1", "ins2", "type", "symmetry_operation1", "symmetry_operation2", "C1_C1_euler_phi", "C1_C1_euler_theta", "C1_C1_euler_psi", "C1_C1_euler_phicospsi", "x1", "x2", "y1", "y2", "z1", "z2", "rmsd_edge_C1N_frame", "label", "pair_bases" ]

    function getTableRows(tuple: object | null) {
        if (!tuple)  return []
        const meta = metadata.find(m => m.pair_type[0].toUpperCase() == pair.id.pairingType[0].toUpperCase() && m.pair_type[1] == pair.id.pairingType[1])

        return Object.entries(tuple).filter(([colName, value]) => !hideColumn(colName, meta, value) && !columnBlacklist.includes(colName)).map(([colName, value]) => {
            const [ label, tooltip ] = getColumnLabel(colName, meta) ?? [ null, null ]
            const quantile = tuple[colName + "_quantile"]
            const logLikelihood = tuple[colName + "_log_likelihood"]
            const modeDeviation = tuple[colName + "_mode_deviation"]
            return { colName, label, tooltip, value, quantile, logLikelihood, modeDeviation }
        })
    }
</script>

<style>
    .imgpane {
        display: flex;
        align-items: center;
    }
    .imgpane > * {
        flex-grow: 1;
        width: 40vw;
    }
    table th {
        white-space: nowrap;
    }

    code {
        color: black;
        background-color: transparent;
    }

    .filter-pass {
        color: #006d09;
        font-weight: bold;
    }

    .filter-fail {
        color: rgb(172, 0, 0);
        font-weight: bold;
        text-decoration: underline;
    }

    .molstar-container {
        width: 70%;
        height: 70vh;
        position: relative;
    }

    .quantile {
        font-weight: bold;
    }
    .quantile.quantile-low {
        color: rgb(172, 0, 0);
    }
    .quantile.quantile-med {
        color: rgb(143, 105, 0);
    }
    .quantile.quantile-high {
        color: #006d09;
    }

</style>
<div>
    <div class="imgpane">
        {#if pair != null && (molStar || allowMolstar && ((imgError && rotError) || (imageUrl == null && videoUrl == null)))}
            <div class="molstar-container">
                <MolStarMaybe pairId={pair.id} />
            </div>
        {:else}
            {#if !avifError}
            <img src={imageUrl.replace(/[.]\w+$/, '-1440.avif')} alt='x' on:error={() => { avifError = true }} />
            {:else if !imgError}
            <img src={imageUrl} alt='The basepair in plane with the screen' on:error={_ => { imgError = true }} />
            {:else}
                <div>
                    <div>Image was not pre-generated </div>
                    <button class="button is-link" on:click={() => { molStar = true }} style="margin-bottom: 1rem">
                        Open the structure in browser
                    </button>
                </div>
            {/if}

            {#if videoUrl && !videoError}
                <video src={videoUrl} autoplay muted loop controls on:error={() => { videoError = true }} />
            {:else if rotImageUrl && !avifRotError}
                <img src={rotImageUrl.replace(/[.]\w+$/, '-1440.avif')} alt='The basepair rotated 90° along the X axis' on:error={_ => {avifRotError=true}} />
            {:else if rotImageUrl && !rotError}
                <img src={rotImageUrl} alt='The basepair rotated 90° along the X axis' on:error={_ => {rotError = true}} />
            {:else}
                <p>Rotated image nor movie was not pre-generated</p>
            {/if}
        {/if}
    </div>
    {#if !molStar && !imgError}
    <div class="control" style="float:right">
        <button class="button" on:click={() => { molStar = true }}>Open the structure in browser</button>
    </div>
    {/if}
    <div>
        <h4>PyMol script<span style="font-size: 1rem; font-weight: 400"> &nbsp;&nbsp;- copy and paste into PyMol command line</span></h4>
        <pre on:click={ev => {
            const selection = window.getSelection()
            selection.removeAllRanges()
            const range = document.createRange()
            range.selectNodeContents(ev.currentTarget)
            selection.addRange(range)
        }}>{generatePymolScript(pair.id).join("\n")}</pre>
    </div>
    <div style="display: flex; flex-direction: row; flex-wrap: wrap; justify-content: space-evenly; gap: 2rem; align-items: flex-start;">
    {#if pairDB?.hbonds}
        <table class="table is-narrow is-striped" style="width: fit-content">
            <tr>
                <th></th>
                {#each pairDB.hbonds as hb, ix}
                    <th>{hb.label ?? `H-Bond ${ix}`}</th>
                {/each}
            </tr>
            <tr>
                <th>Length</th>
                {#each pairDB.hbonds as hb, i}
                    {@const range = realFilter?.bond_length?.[i]}
                    <td class:filter-pass={isOutOfRange(hb.length, range) === true}
                        class:filter-fail={isOutOfRange(hb.length, range) === false}
                        title={getRangeValueTitle(hb.length, range)}>
                        {hb.length == null ? 'NULL' : hb.length?.toFixed(2) + " Å"}</td>
                {/each}
                {#if realFilter?.min_bond_length?.max != null && pairDB.hbonds.every(hb => hb.length == null || hb.length > realFilter.min_bond_length.max)}
                    <td class="filter-fail" title="All bonds are shorter than the minimum allowed length">
                        <i>All ≥ {realFilter.min_bond_length.max}</i>
                    </td>
                {/if}
            </tr>
            <tr>
                <th>Donor angle</th>
                {#each pairDB.hbonds as hb, i}
                    {@const range = realFilter?.bond_donor_angle?.[i]}
                    <td class:filter-pass={isOutOfRange(hb.donorAngle, range) === true}
                        class:filter-fail={isOutOfRange(hb.donorAngle, range) === false}
                        title={getRangeValueTitle(hb.donorAngle, range)}>
                        {hb.donorAngle == null ? 'NULL' : hb.donorAngle?.toFixed(0)+"°"}
                    </td>
                {/each}
            </tr>
            <tr>
                <th>Acceptor angle</th>
                {#each pairDB.hbonds as hb, i}
                    {@const range = realFilter?.bond_acceptor_angle?.[i]}
                    <td class:filter-pass={isOutOfRange(hb.acceptorAngle, range) === true}
                        class:filter-fail={isOutOfRange(hb.acceptorAngle, range) === false}
                        title={getRangeValueTitle(hb.acceptorAngle, range)}>
                        {hb.acceptorAngle == null ? 'NULL' : hb.acceptorAngle?.toFixed(0)+"°"}
                    </td>
                {/each}
            </tr>
            <tr>
                <th>Angle to left plane</th>
                {#each pairDB.hbonds as hb, i}
                    {@const range = realFilter?.bond_plane_angle1?.[i]}
                    <td class:filter-pass={isOutOfRange(hb.OOPA1, range) === true}
                        class:filter-fail={isOutOfRange(hb.OOPA1, range) === false}
                        title={getRangeValueTitle(hb.OOPA1, range)}>
                        {hb.OOPA1 == null ? 'NULL' : hb.OOPA1?.toFixed(0)+"°"}</td>
                {/each}
            </tr>
            <tr>
                <th>Angle to right plane</th>
                {#each pairDB.hbonds as hb, i}
                    {@const range = realFilter?.bond_plane_angle2?.[i]}
                    <td class:filter-pass={isOutOfRange(hb.OOPA2, range) === true}
                        class:filter-fail={isOutOfRange(hb.OOPA2, range) === false}
                        title={getRangeValueTitle(hb.OOPA2, range)}>
                        {hb.OOPA2 == null ? 'NULL' : hb.OOPA2?.toFixed(0)+"°"}
                    </td>
                {/each}
            </tr>
        </table>
    {/if}
    {#if pairDB?.originalRow?.quantile_mean_Q != null}
        {@const r = pairDB.originalRow}
        {@const score = !['quantile_mean_Q', 'quantile_mean_QA'].includes(realFilter.orderBy) && r.quantile_hmean_Q != null ? r.quantile_hmean_Q : r.quantile_mean_Q}
        <table class="table is-narrow is-striped" style="width: fit-content">
            <tr>
                <th>Probability percentile</th>
                <td class="quantile {score < 0.3 ? "quantile-low" : score < 0.7 ? "quantile-med" : "quantile-high"}"
                    title="Harmonic mean percentile = {r.quantile_hmean_Q*100}, arithmetic mean percentile = {r.quantile_mean_Q*100}">
                    {(score*100).toFixed(1)}%
                </td>
            </tr>
            <tr>
                <th>Percentile mean</th>
                <td>{(r.quantile_mean*100).toFixed(1)}%</td>
            </tr>
            {#if r.quantile_hmean != null}
            <tr>
                <th>Percentile harmonic mean</th>
                <td title="1/∑(1/pₖ) = {r.quantile_hmean*100}">{(r.quantile_hmean*100).toFixed(1)}%</td>
            </tr>
            {/if}
            <tr>
                <th>Percentile min</th>
                <td>{(r.quantile_min*100).toFixed(1)}%</td>
            </tr>
            <tr>
                <th>Percentile second min</th>
                <td>{(r.quantile_min2*100).toFixed(1)}%</td>
            </tr>
        </table>
    {/if}
    {#if pair?.id?.nt1?.pdbid}
    <div>
        <p>
            {_.capitalize(longNucleotideNames[pair.id.nt1.resname] ?? pair.id.nt1.resname)} <strong>{pair.id.nt1.resnum}{pair.id.nt1.inscode ? '.' + pair.id.nt1.inscode : ''}</strong> in chain <strong>{pair.id.nt1.chain}</strong>
            forms {pair.id.pairingType[0]} basepair with
            {_.capitalize(longNucleotideNames[pair.id.nt2.resname] ?? pair.id.nt2.resname)} <strong>{pair.id.nt2.resnum}{pair.id.nt2.inscode ? '.' + pair.id.nt2.inscode : ''}</strong> in chain <strong>{pair.id.nt2.chain}</strong>
        </p>
        <h5 style="margin-bottom: 0px">From structure{[null, undefined, '', 0, 1, '1', '0'].includes(pair.id.nt1.model) ? '' : ` (model ${pair.id.nt1.model})`}</h5>
        <div class='media' style="max-width: 600px">
            <div class="media-left">
                <a href="https://www.rcsb.org/structure/{pair.id.nt1.pdbid}">
                <code style="font-size: 3rem">{pair.id.nt1.pdbid}</code>
                </a>
            </div>
            <div class="media-content">
            <div class="content">
                <p>
                <strong>{pairDB?.originalRow?.structure_method ?? ''}</strong> <small> at {pairDB?.originalRow?.resolution?.toFixed(2) ?? '?'} Å</small> <small>(published {pairDB?.originalRow?.deposition_date ? new Date(pairDB?.originalRow.deposition_date).toLocaleDateString('en', {month: 'long', day: 'numeric',  year: 'numeric'}) : ''})</small>
                <br>
                {pairDB?.originalRow?.structure_name ?? ''}
                </p>
            </div>
            </div>
        </div>
    </div>
    {/if}
    <!-- <table class="table is-narrow is-striped" style="width: fit-content">
        <tr>
            <th>Structure ID</th>
            <td>{pairDB?.id.nt1.pdbid}</td>
        </tr>
        <tr>
            <th>Structure Name</th>
            <td>{pairDB?.originalRow?.structure_name ?? ''}</td>
        </tr>
        <tr>
            <th>Structure Method</th>
            <td>{pairDB?.originalRow?.structure_method ?? ''}</td>
        </tr>
        <tr>
            <th>Resolution </th>
            <td>{pairDB?.originalRow?.resolution ?? '?'} Å</td>
        </tr>
        <tr>
            <th>Deposition date</th>
            <td>{pairDB?.originalRow?.deposition_date ? new Date(pairDB?.originalRow.deposition_date).toLocaleDateString('en', {month: 'long', day: 'numeric',  year: 'numeric'}) : ''}</td>
        </tr>
    </table> -->
    </div>
    <div style="font-style: italic;">
        {#if true}
        {@const conditions = filterToSqlCondition(realFilter).filter(c => c !='jirka_approves')}
        {#if filterLabel}
            <p style="margin-bottom: 1rem">Implicit filter: <a href={config.parameterBoundariesUrls[filterLabel]}><strong>{filterLabel}</strong></a> with {conditions.length} conditions.</p>
        {:else if conditions.length > 0}
            <p style="margin-bottom: 1rem" title={conditions.join("\n AND ")}>Filtered by {conditions.length} conditions.</p>
        {/if}
        {/if}
    </div>
    <div>
        <table class="table is-narrow is-striped" style="width: fit-content">
            
        <tbody>
        {#each getTableRows(pairDB?.originalRow) as r}
            {@const filterRange = getRange(realFilter, r.colName)}
            {@const val = r.value}
            <tr>
                <td><b><code>{r.colName}</code></b></td>
                <td>{r.label ?? ''}</td>
                <td colspan={['structure_name', 'structure_method', 'pairid', 'deposition_date'].includes(r.colName) ? 3 : 1}
                    style="font-weigth: 700; text-align: {[ "bigint", "number", "boolean" ].includes(typeof r.value) ? 'right' : 'left'};"
                    data-debug-filter-range={JSON.stringify(filterRange)}
                    data-debug-nofilter={filterRange == null}
                    class:filter-pass={isOutOfRange(val, filterRange) === true}
                    class:filter-fail={(val == null && filterRange != null) || isOutOfRange(val, filterRange) === false}
                    title={getRangeValueTitle(val, filterRange)}
                    data-type={typeof val}>
                    {typeof val == "number" ? val.toFixed(3) : val == null ? (filterRange != null ? "NULL" : "") : "" + val}
                </td>
                <td>
                    {#if r.quantile != null}
                        <span title="Probability likelihood percentile" class="quantile {r.quantile < 0.3 ? "quantile-low" : r.quantile < 0.7 ? "quantile-med" : "quantile-high"}">{(r.quantile*100).toFixed(1)}%</span>
                    {/if}
                </td>
                <td><i>{r.tooltip ?? ''}</i></td>
            </tr>
        {/each}
        </tbody>
        </table>
        <!-- <pre>{JSON.stringify(pair, null, 2)}</pre> -->
    </div>
</div>
