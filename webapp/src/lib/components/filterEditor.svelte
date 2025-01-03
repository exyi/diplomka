<script lang="ts">
	import { defaultFilter, filterToSqlCondition, getDataSourceTable, makeSqlQuery, type ComparisonMode, type NucleotideFilterModel, type NumRange, orderByOptions, hasFilters } from "$lib/dbModels";
  import type metadataModule from '$lib/metadata';
  import RangeSlider from 'svelte-range-slider-pips'
  import * as filterLoader from '$lib/predefinedFilterLoader'
	import RangeEditor from "./RangeEditor.svelte";
	import _, { slice, startsWith } from "lodash";
	import config from "$lib/config";
	import Pairimage from "./pairimage.svelte";
	import { imgDir } from "$lib/dbInstance";
	import ClearFilterButton from "./ClearFilterButton.svelte";
	import DownloadButton from "./DownloadButton.svelte";

    export let filter: NucleotideFilterModel
    export let filterBaseline: NucleotideFilterModel | undefined
    export let allowFilterBaseline: boolean = true
    export let comparisonMode: ComparisonMode | undefined = undefined
    export let selectingFromTable: string | null = null
    export let metadata: typeof metadataModule[0] | null = null
    export let mode: "gui" | "sql" | "basic" = "basic"

    let bonds = ["Bond 0", "Bond 1", "Bond 2"]
    $: bonds = metadata?.labels?.filter(l => l != null) ?? ["Bond 0", "Bond 1", "Bond 2"]
    function tryParseNum(x: string) {
      const n = Number.parseFloat(x)
      return isNaN(n) ? null : n
    }

    function setBaseline(f: NucleotideFilterModel | undefined, mode: string) {
      if (mode == "sql" || f == null) {
        filterBaseline = f
      } else {
        filterBaseline = { ...f, sql: undefined }
      }
    }

    function formatFilter(f: NucleotideFilterModel, compareWith: NucleotideFilterModel | undefined) {
      if (f == null) return ""
      const clauses = filterToSqlCondition(f).filter(x => !["jirka_approves"].includes(x))
      const currentClauses = filterToSqlCondition(compareWith).filter(x => !["jirka_approves"].includes(x))
      const newClauses = clauses.filter(x => !currentClauses.includes(x))
      const removedClauses = currentClauses.filter(x => !clauses.includes(x))
      const datasetName = f.datasource == "fr3d-f" ? "FR3D, Representative Set" :
                          f.datasource == "fr3d" ? "FR3D, entire PDB" :
                          f.datasource == "fr3d-nf" ? "FR3D with nears, RS" :
                          f.datasource == "fr3d-n" ? "FR3D with nears, PDB" :
                          f.datasource == "allcontacts-f" ? "All polar contacts, RS" :
                          f.datasource == "allcontacts" ? "All polar contacts, PDB" :
                          f.datasource == "allcontacts-boundaries-f" ? "Pairs Selected by New Parameters, RS" :
                          f.datasource == "allcontacts-boundaries" ? "Pairs Selected by New Parameters, PDB" :
                          "dataset???"
      const sameDataSource = compareWith?.datasource == f.datasource
      const sameFilters = newClauses.length == 0 && removedClauses.length == 0
      if (sameDataSource && sameFilters) return "Same data"
      if (sameDataSource && clauses.length == 0) return "Same data, without filters"
      if (sameDataSource && newClauses.length == 0) return `Same data, without ${removedClauses.length} filters`
      if (sameDataSource && removedClauses.length == 0) return `Same data, with ${newClauses.length} more filters`
      if (sameDataSource) return `Same data, with other filters`
      if (newClauses.length == 0) return datasetName

      return `${datasetName} with other filters`
    }

    function ensureLength(array: NumRange[] | any, index: number) {
      while (array.length <= index) {
        array.push({})
      }
    }

    function modeChange(e: Event & { currentTarget:EventTarget & HTMLInputElement }) {
      mode = e.currentTarget.value as any

      const currentSqlQuery = makeSqlQuery(selectingFromTable.endsWith('_f') ? {... filter, filtered: false} : filter, selectingFromTable)
      if (mode == "sql" && !filter.sql) {
        filter = {...filter, sql: currentSqlQuery }
      }
      if (mode != "sql" && filter?.sql?.trim() == currentSqlQuery.trim()) {
        filter = {...filter, sql: "" }
      }
    }

    function dnaRnaChange(e: Event & { currentTarget:EventTarget & HTMLInputElement }) {
      const v = {
        "rna": false,
        "dna": true,
        "both": undefined
      }[e.currentTarget.value as any]
      filter = {...filter, dna: v }
    }

    function getOrderByOptions(currentOption: string, allowed: string[] | null) {
      const virtualOpt = orderByOptions.some(x => x.id == currentOption) ? [] : [{ id: currentOption, expr: currentOption, label: _.truncate(currentOption, {length: 60}), title: "Custom sort expression " + currentOption }]
      if (allowed == null) {
        return [...orderByOptions, ...virtualOpt ]
      }
      else {
        return [...orderByOptions.filter(x => allowed.includes(x.id) || x.id == currentOption), ...virtualOpt ]
      }
    }

    async function setFr3dObservedBoundaries(sql: boolean) {
      const f = (await filterLoader.defaultFilterLimits.value).limits
      const pairType = metadata.pair_type.join("-")
      const newFilter = filterLoader.addHBondLengthLimits(pairType, 0.0, filterLoader.toNtFilter(f, 0.0, pairType, null))
      newFilter.datasource = filter.datasource
      newFilter.filtered = filter.filtered && !["fr3d-f", "allcontacts-f", "allcontacts-boundaries-f"].includes(filter.datasource)
      if (sql) {
        newFilter.sql = makeSqlQuery(newFilter, getDataSourceTable(newFilter))
      }
      newFilter.filtered = filter.filtered
      newFilter.rotX = filter.rotX
      newFilter.orderBy = filter.orderBy
      newFilter.dna = filter.dna
      newFilter.resolution = filter.resolution
      filter = newFilter
    }

    let hasYawPitchRoll = false
    $: hasYawPitchRoll = Boolean(filter.yaw1 || filter.pitch1 || filter.roll1 || filter.yaw2 || filter.pitch2 || filter.roll2)

    function dataSourceChange(newDS: string) {
      if (newDS == null) {
        newDS = config.defaultDataSource
      }
      const filtered = newDS.endsWith('-f') || newDS.endsWith('-nf')
      filter = {...filter, datasource: newDS as any, filtered }
    }

    function selectInput(ev: Event & { currentTarget: HTMLInputElement }) {
      ev.currentTarget.select()
    }

    let hb_params: {k: keyof NucleotideFilterModel, name: string, title: string, step?: number, min?: number, max?: number }[]
    
    $: {
      hb_params = [
        { k: "bond_length", name: "Length", title: "Length in Å between the donor and acceptor heavy atoms", step: 0.1, min: 0, max: 6 },
        { k: "bond_acceptor_angle", name: "Acceptor Angle", title: "Angle in degrees between the acceptor, donor and its covalently bound atom.", step: 5, min: 0, max: 180 },
        { k: "bond_donor_angle", name: "Donor Angle", title: "Angle in degrees between the donor, acceptor and its covalently bound atom", step: 5, min: 0, max: 180 },
        { k: "bond_plane_angle1", name: "Left Plane Angle", title: "Angle of the left plane with the line connecting the heavy atoms", step: 5, min: -90, max: 90 },
        { k: "bond_plane_angle2", name: "Right Plane Angle", title: "Angle of the right plane with the line connecting the heavy atoms", step: 5, min: -90, max: 90 }
      ]
      hb_params = hb_params.slice(0, filter.bond_plane_angle1.length + filter.bond_plane_angle2.length > 0 ? 5 : 3)
    }
</script>

<style>
  .panel-title {
    font-variant: small-caps;
    font-size: 1rem;
    font-weight: bold;
    text-align: center;
    height: 1.5rem;
  }
  .panel-field {
    height: 1.75rem;
    margin-bottom: 0.75rem;
  }
  .flex-columns {
    display: flex;
    width: 100%;
    justify-content: center;
    flex-wrap: wrap;
  }
  .flex-columns > * {
    flex: 0 1 auto;
  }

  .num-input {
    max-width: 100px;
  }

  .field-body {
    flex-grow: 1;
  }

  .mode-selection {
    display: flex;
    justify-content: center;
  }
  .sql-editor {
    font-family: 'Fira Code', 'Consolas', monospace;
  }
</style>

<div>
    {#if mode != 'basic'}
    <div class="control mode-selection">
        <label class="radio" title="Filter by constraining the H-bond parameters.">
          <input type="radio" checked={mode=="basic"} value="basic" name="editor_mode" on:change={modeChange}>
          Standard
        </label>
        <label class="radio" title="Filter by constraining the H-bond parameters.">
          <input type="radio" checked={mode=="gui"} value="gui" name="editor_mode" on:change={modeChange}>
          Dev GUI
        </label>
        <label class="radio" title="Filter by anything you want using SQL.">
          <input type="radio" checked={mode=="sql"} value="sql" name="editor_mode" on:change={modeChange}>
          Dev SQL
        </label>
    </div>
    {/if}

    {#if mode=="basic"}
    {@const biggestStat = metadata?.statistics.reduce((prev, x) => (prev && prev.count > x.count ? prev : x), null)}
    {@const igmUrlBase = biggestStat?.nicest_bp == null ? null : `${imgDir}/${biggestStat.nicest_bp[0]}/${biggestStat.nicest_bp[2]}_${biggestStat.nicest_bp[4]}${biggestStat.nicest_bp[5]??''}${biggestStat.nicest_bp[6]??''}-${biggestStat.nicest_bp[7]}_${biggestStat.nicest_bp[9]}${biggestStat.nicest_bp[10]??''}${biggestStat.nicest_bp[11]??''}`}
    <div class="flex-columns">
      {#if biggestStat}
      <div class="column">
        <div class="LW-table-cell" style="text-align: center">
          <!-- parentSize={true} -->
              <Pairimage
                url={igmUrlBase + ".png"}
                videoUrl={igmUrlBase + ".webm"}
                allowHoverVideo={false}
                onClick={() => false}
                labelText="Reference {metadata.pair_type[0]} {metadata.pair_type[1]} basepair"
                pair={JSON.parse(JSON.stringify({ // TS laundry machine
                    id: {
                        nt1: {pdbid: biggestStat.nicest_bp[0], model: biggestStat.nicest_bp[1], chain: biggestStat.nicest_bp[2], resname: biggestStat.nicest_bp[3], resnum: biggestStat.nicest_bp[4], altloc: biggestStat.nicest_bp[5], inscode: biggestStat.nicest_bp[6]},
                        nt2: {pdbid: biggestStat.nicest_bp[0], model: biggestStat.nicest_bp[1], chain: biggestStat.nicest_bp[7], resname: biggestStat.nicest_bp[8], resnum: biggestStat.nicest_bp[9], altloc: biggestStat.nicest_bp[10], inscode: biggestStat.nicest_bp[11]},
                        pairingType: metadata.pair_type
                    }
                }))} />
        </div>
      </div>
      {/if}
      <div class="column">
        <div class="field">
          <label class="label" for="ntfilter-data-source">Data source</label>
          <div class="control">
            <div class="select">
              <select
                value={filter.datasource ?? config.defaultDataSource}
                id="ntfilter-data-source"
                on:change={ev => {
                  dataSourceChange(ev.currentTarget.value)
                }}>
                <option value="fr3d-f" disabled={config.disabledDataSources.includes("fr3d-f")}>FR3D — Reference Set</option>
                <option value="fr3d" disabled={config.disabledDataSources.includes("fr3d")}>FR3D — Entire PDB</option>
                <!-- <option value="fr3d-nf">FR3D with nears, RS</option>
                <option value="fr3d-n">FR3D with nears, PDB</option> -->
                <option value="allcontacts-f" disabled={config.disabledDataSources.includes("allcontacts-f")} title="All polar contacts reminiscent of the basepair - nucleotide pair with <= 4 Å between any polar atoms, <= 4.2 Å on at least one defined H-bond, and <= 2.5 Å edge to plane distance">All Polar Contacts — Reference Set</option>
                <option value="allcontacts-boundaries-f" disabled={config.disabledDataSources.includes("allcontacts-boundaries-f")}>Pairs Selected by New Parameters</option>
              </select>
            </div>
          </div>
        </div>
        <div class="control">
          <label class="radio" title="At least one of the nucleotides is RNA">
            <input type="radio" name="rna_dna_mode" value="rna" checked={filter.dna == false} on:change={dnaRnaChange}>
            RNA
          </label>
          <label class="radio" title="At least one of the nucleotides is DNA">
            <input type="radio" name="rna_dna_mode" value="dna" checked={filter.dna == true} on:change={dnaRnaChange}>
            DNA
          </label>
          <label class="radio">
            <input type="radio" name="rna_dna_mode" value="both" checked={filter.dna == null} on:change={dnaRnaChange}>
            Both
          </label>
        </div>

      </div>
      <div class="column">
        <div class="field">
          <label class="label" for="ntfilter-order-by">Order by</label>
          <div class="control">
            <div class="select">
              <select bind:value={filter.orderBy} id="ntfilter-order-by">
                {#each getOrderByOptions(filter.orderBy ?? '', ["pdbid", "rmsdA", "rmsdD", "quantile_hmean_Q", "quantile_hmean_QA"]) as opt}
                  <option value={opt.id} title={opt.title}>{opt.label}</option>
                {/each}
              </select>
            </div>
          </div>
        </div>

        <div class="control">
          <label class="checkbox" title="Rotate images 90° along X-axis to see the coplanarity">
            <input type="checkbox" checked={!!filter.rotX} on:change={e => filter = {...filter, rotX: e.currentTarget.checked }}>
            Rotate images
          </label>
        </div>
      </div>
      <div class="column" style="display: flex; flex-direction: column; justify-content: center">
        <button class="button" class:is-warning={hasFilters(filter, mode)} disabled={!hasFilters(filter, mode)} on:click={() => { filter = defaultFilter() } }>Reset filters</button>

        {#if allowFilterBaseline && filterBaseline == null}
          {#if filter.datasource?.startsWith("allcontacts")}
            <button class="button" type="button" on:click={() => setBaseline({ ...defaultFilter(), datasource: filter.filtered ? "fr3d-f" : "fr3d" }, "gui")}>
              Enable FR3D comparison
            </button>
          {:else}
            <!-- <button class="button" type="button" on:click={() => setBaseline(structuredClone(filter), mode)}>
              Compare with ???
            </button> -->
          {/if}
        {:else if filterBaseline != null}
          {#if comparisonMode != null}
          <div class="select">
            <select bind:value={comparisonMode}>
              <option value="union">Show all matches</option>
              <option value="difference">Only differences</option>
              <option value="new">Absent in {filterBaseline.datasource?.startsWith("fr3d") && !filter.datasource?.startsWith("fr3d") ? "FR3D" : "baseline"} (green)</option>
              <option value="missing">Only in {filterBaseline.datasource?.startsWith("fr3d") && !filter.datasource?.startsWith("fr3d") ? "FR3D" : "baseline"} (red)</option>
            </select>
          </div>
          {/if}
          
          <button class="button is-warning" type="button" on:click={() => setBaseline(null, mode)}>
            Exit comparison
          </button>
        {/if}
      </div>
    </div>
    
    {:else if mode=="gui"}
    <div class="flex-columns" >
        <div class="column" style="font-variant: small-caps; font-weight: bold">
          <div class="panel-title"></div>
          {#each hb_params as p}
            <div class="panel-field" title={p.title}>{p.name} <ClearFilterButton bind:clearList={filter[p.k]} /></div>
          {/each}
          {#if hasYawPitchRoll}
            <div class="panel-title"></div>
            <div class="panel-field">Left to right <ClearFilterButton bind:clear1={filter.yaw1} bind:clear2={filter.pitch1} bind:clear3={filter.roll1} /></div>
            <div class="panel-field">Right to left <ClearFilterButton bind:clear1={filter.yaw2} bind:clear2={filter.pitch2} bind:clear3={filter.roll2} /></div>
          {/if}
        </div>
        <div class="column">
            <h3 class="panel-title">{bonds[0]}</h3>
            {#each hb_params as p}
              {@const i = 0}
              <div class="panel-field field is-horizontal">
                <div class="field-body">
                  <div class="field">
                    <div class="field has-addons">
                      <p class="control">
                        <input class="input is-small num-input" type="number" step={p.step} min={p.min} max={p.max} placeholder="Min" value={filter[p.k][i]?.min} on:change={ev => { ensureLength(filter[p.k], i); filter[p.k][i].min = tryParseNum(ev.currentTarget.value)} } on:click={selectInput}>
                      </p>
                      <p class="control">
                        <input class="input is-small num-input" type="number" step={p.step} min={p.min} max={p.max} placeholder="Max" value={filter[p.k][i]?.max} on:change={ev => { ensureLength(filter[p.k], i); filter[p.k][i].max = tryParseNum(ev.currentTarget.value)} } on:click={selectInput}>
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            {/each}
            {#if hasYawPitchRoll}
              <h3 class="panel-title">Yaw <ClearFilterButton bind:clear1={filter.yaw1} bind:clear2={filter.yaw2} /></h3>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.yaw1} step={5} min={-180} max={180} />
              </div>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.yaw2} step={5} min={-180} max={180} />
              </div>
            {/if}

            {#if filter.coplanarity_angle}
              <h3 class="panel-title">Coplanarity angle <ClearFilterButton bind:clear1={filter.coplanarity_angle} /></h3>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.coplanarity_angle} step={5} min={-180} max={180} />
              </div>
            {/if}
            {#if filter.min_bond_length}
              <h3 class="panel-title" title="At least one H-bond must satisfy this length constraint">Min H-bond length <ClearFilterButton bind:clear1={filter.min_bond_length}/></h3>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.min_bond_length} step={0.1} min={0} max={4.2} />
              </div>
            {/if}
        </div>
        <div class="column">
            <h3 class="panel-title">{bonds[1] ?? ''}</h3>
            {#each hb_params as p}
              {@const i = 1}
              <div class="panel-field field is-horizontal">
                {#if bonds[1] != null}
                <div class="field-body">
                  <div class="field">
                    <div class="field has-addons">
                      <p class="control">
                        <input class="input is-small num-input" type="number" step={p.step} min={p.min} max={p.max} placeholder="Min" value={filter[p.k][i]?.min} on:change={ev => { ensureLength(filter[p.k], i); filter[p.k][i].min = tryParseNum(ev.currentTarget.value)} } on:click={selectInput}>
                      </p>
                      <p class="control">
                        <input class="input is-small num-input" type="number" step={p.step} min={p.min} max={p.max} placeholder="Max" value={filter[p.k][i]?.max} on:change={ev => { ensureLength(filter[p.k], i); filter[p.k][i].max = tryParseNum(ev.currentTarget.value)} } on:click={selectInput}>
                      </p>
                    </div>
                  </div>
                </div>
                {/if}
              </div>
            {/each}
            {#if hasYawPitchRoll}
              <h3 class="panel-title">Pitch <ClearFilterButton bind:clear1={filter.pitch1} bind:clear2={filter.pitch2}/></h3>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.pitch1} step={5} min={-180} max={180} />
              </div>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.pitch2} step={5} min={-180} max={180} />
              </div>
            {/if}

            {#if filter.coplanarity_edge_angle1}
              <h3 class="panel-title">Edge1/Plane2 angle <ClearFilterButton bind:clear1={filter.coplanarity_edge_angle1}/></h3>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.coplanarity_edge_angle1} step={5} min={-180} max={180} />
              </div>
            {/if}

            {#if filter.coplanarity_shift1}
              <h3 class="panel-title">Edge1/Plane2 distance <ClearFilterButton bind:clear1={filter.coplanarity_shift1}/></h3>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.coplanarity_shift1} step={0.1} min={-2.5} max={2.5} />
              </div>
            {/if}
        </div>

        <div class="column">
            <h3 class="panel-title">{bonds[2] ?? ''}</h3>
            {#each hb_params as p}
              {@const i = 2}
              <div class="panel-field field is-horizontal">
                {#if bonds[2] != null}
                <div class="field-body">
                  <div class="field">
                    <div class="field has-addons">
                      <p class="control">
                        <input class="input is-small num-input" type="number" step={p.step} min={p.min} max={p.max} placeholder="Min" value={filter[p.k][i]?.min} on:change={ev => { ensureLength(filter[p.k], i); filter[p.k][i].min = tryParseNum(ev.currentTarget.value)} } on:click={selectInput}>
                      </p>
                      <p class="control">
                        <input class="input is-small num-input" type="number" step={p.step} min={p.min} max={p.max} placeholder="Max" value={filter[p.k][i]?.max} on:change={ev => { ensureLength(filter[p.k], i); filter[p.k][i].max = tryParseNum(ev.currentTarget.value)} } on:click={selectInput}>
                      </p>
                    </div>
                  </div>
                </div>
                {/if}
              </div>
            {/each}
            {#if hasYawPitchRoll}
              <h3 class="panel-title">Roll <ClearFilterButton bind:clear1={filter.roll1} bind:clear2={filter.roll2}/></h3>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.roll1} step={5} min={-180} max={180} />
              </div>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.roll2} step={5} min={-180} max={180} />
              </div>
            {/if}

            {#if filter.coplanarity_edge_angle2}
              <h3 class="panel-title">Edge2/Plane1 angle <ClearFilterButton bind:clear1={filter.coplanarity_edge_angle2}/></h3>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.coplanarity_edge_angle2} step={5} min={-180} max={180} />
              </div>
            {/if}
            {#if filter.coplanarity_shift2}
              <h3 class="panel-title">Edge2/Plane1 distance <ClearFilterButton bind:clear1={filter.coplanarity_shift2}/></h3>
              <div class="panel-field field is-horizontal">
                <RangeEditor bind:range={filter.coplanarity_shift2} step={0.1} min={-2.5} max={2.5} />
              </div>
            {/if}
        </div>

        {#each bonds.slice(3) as additional_bond, rel_i}
        <div class="column">
          <h3 class="panel-title">{additional_bond}</h3>
          {#each hb_params as p}
            {@const i = 3 + rel_i}
            <div class="panel-field field is-horizontal">
              <div class="field-body">
                <div class="field">
                  <div class="field has-addons">
                    <p class="control">
                      <input class="input is-small num-input" type="number" step={p.step} min={p.min} max={p.max} placeholder="Min" value={filter[p.k][i]?.min} on:change={ev => { ensureLength(filter[p.k], i); filter[p.k][i].min = tryParseNum(ev.currentTarget.value)} } on:click={selectInput}>
                    </p>
                    <p class="control">
                      <input class="input is-small num-input" type="number" step={p.step} min={p.min} max={p.max} placeholder="Max" value={filter[p.k][i]?.max} on:change={ev => { ensureLength(filter[p.k], i); filter[p.k][i].max = tryParseNum(ev.currentTarget.value)} } on:click={selectInput}>
                    </p>
                  </div>
                </div>
              </div>
            </div>
          {/each}
          </div>
        {/each}

        <div class="column">
          <div class="control">
            <label class="radio" title="At least one of the nucleotides is RNA">
              <input type="radio" name="rna_dna_mode" value="rna" checked={filter.dna == false} on:change={dnaRnaChange}>
              RNA
            </label>
            <label class="radio" title="At least one of the nucleotides is DNA">
              <input type="radio" name="rna_dna_mode" value="dna" checked={filter.dna == true} on:change={dnaRnaChange}>
              DNA
            </label>
            <label class="radio">
              <input type="radio" name="rna_dna_mode" value="both" checked={filter.dna == null} on:change={dnaRnaChange}>
              Both
            </label>
          </div>

          <div class="field">
            <label class="label" for="ntfilter-data-source">Data source</label>
            <div class="control">
              <div class="select is-small">
                <select
                  value={filter.datasource ?? config.defaultDataSource}
                  id="ntfilter-data-source"
                  on:change={ev => {
                    dataSourceChange(ev.currentTarget.value)
                  }}>
                  <option value="fr3d-f" disabled={config.disabledDataSources.includes("fr3d-f")}>FR3D, Representative Set</option>
                  <option value="fr3d" disabled={config.disabledDataSources.includes("fr3d")}>FR3D, Entire PDB</option>
                  <option value="fr3d-nf" disabled={config.disabledDataSources.includes("fr3d-nf")}>FR3D with nears, RS</option>
                  <option value="fr3d-n" disabled={config.disabledDataSources.includes("fr3d-n")}>FR3D with nears, PDB</option>
                  <option value="allcontacts-f" disabled={config.disabledDataSources.includes("allcontacts-f")}>All Polar Contacts, RS</option>
                  <option value="allcontacts" disabled={config.disabledDataSources.includes("allcontacts")}>All Polar Contacts, PDB</option>
                  <option value="allcontacts-boundaries-f" disabled={config.disabledDataSources.includes("allcontacts-boundaries-f")}>Pairs Selected by New Parameters, RS</option>
                  <option value="allcontacts-boundaries" disabled={config.disabledDataSources.includes("allcontacts-boundaries")}>Pairs Selected by New Parameters, PDB</option>
                </select>
              </div>
            </div>
          </div>
          {#if ["allcontacts-boundaries-f", "allcontacts-boundaries"].includes(filter.datasource)}
            <div class="field">
              <button class="button is-small" on:click={async ()=> {
                await setFr3dObservedBoundaries(false)
                filter = {...filter, datasource: "allcontacts-boundaries" == filter.datasource ? "allcontacts" : "allcontacts-f"}
              } }>Edit the selection boundaries</button>
            </div>
          {/if}
          
          {#if hasFilters(filter, mode)}
          <div class="field">
            <button class="button is-small is-warning" on:click={() => { filter = defaultFilter() } }>Reset filters</button>
          </div>
          {/if}

          {#if [].includes(filter.datasource)}
          <div class="control">
            <label class="checkbox" title="Filter out redundant nucleotides or nucleoties with bad something TODO">
              <input type="checkbox" checked={filter.filtered} on:change={e => filter = {...filter, filtered: e.currentTarget.checked }}>
              Representative set only
            </label>
          </div>
          {/if}

          <div class="field has-addons">
            {#if filter.resolution?.min != null}
              <div class="control">
                <input class="input is-small num-input" style="max-width:4rem" type="number" step="0.1" min=0 max={filter.filtered ? 3.5 : 20} placeholder="Min" value={filter.resolution?.min ?? 0} on:change={ev => { filter.resolution ??= {}; filter.resolution.min = tryParseNum(ev.currentTarget.value)} } on:click={selectInput}>
              </div>
            {/if}
            <label class="label" for="ntfilter-resolution">{#if filter.resolution?.min != null}&nbsp;≤ {/if}Resolution ≤&nbsp;</label>
            <div class="control">
              <input class="input is-small num-input" style="max-width:4rem" type="number" step="0.1" min=0 max={filter.filtered ? 3.5 : 20} placeholder={filter.filtered ? '3.5' : ''} value={filter.resolution?.max ?? ''} on:change={ev => { filter.resolution ??= {}; filter.resolution.max = tryParseNum(ev.currentTarget.value)}} on:click={selectInput}>
            </div>
            &nbsp;Å
          </div>
          {#if filter.filtered && filter.resolution?.max && filter.resolution?.max > 3.5}
            <p class="help is-danger">Representative set only<br> contains structures ≤3.5 Å</p>
          {/if}

          <div class="field">
            <label class="label" for="ntfilter-validation">Probability percentile:</label>
            <div class="control">
              <RangeEditor bind:range={filter.validation_score} step={1} min={0} max={100} />
            </div>
          </div>

        </div>

        <div class="column">
          <div class="field">
            <label class="label" for="ntfilter-order-by">Order by</label>
            <div class="control">
              <div class="select is-small">
                <select bind:value={filter.orderBy} id="ntfilter-order-by">
                  {#each getOrderByOptions(filter.orderBy ?? '', null) as opt}
                    <option value={opt.id} title={opt.title}>{opt.label}</option>
                  {/each}
                </select>
              </div>
            </div>
          </div>

          {#if allowFilterBaseline}
          <div class="field">
            <label class="label" for="ntfilter-order-by">Comparison baseline</label>
            <div class="control">
              {#if filterBaseline == null}
                <div class="buttons has-addons">
                  <button class="button is-small" on:click={() => setBaseline({ ...defaultFilter(), datasource: filter.filtered ? "fr3d-f" : "fr3d", filtered: filter.filtered }, "gui")}
                    title="Sets the current filters as the filter baseline, allowing you to change some parameters and observe the changed">
                    FR3D
                  </button>
                  <button class="button is-small" on:click={() => setBaseline(structuredClone(filter), mode)}
                    title="Compares the current basepair selection with that determined by FR3D">
                    Set to this
                  </button>
                </div>
              {:else}
                <p>{formatFilter(filterBaseline, filter)}</p>
                <div class="buttons has-addons">
                  <button class="button is-small is-warning" type="button" on:click={() => setBaseline(null, mode)}
                    title="Exits comparison mode, removed the baseline">
                    ❌ Reset
                  </button>
                  <button class="button is-small" on:click={() => {
                    const backup = structuredClone(filterBaseline)
                    setBaseline(structuredClone(filter), mode)
                    filter = backup
                  }}
                    title="Swap the baseline and current dataset">
                    ↔️ Swap
                  </button>
                  <!-- <button class="button" type="button" on:click={() => setBaseline(filter, mode)}>
                    Set to this
                  </button> -->
                </div>
                <div class="select is-small">
                  <select bind:value={comparisonMode}>
                    <option value="union">Show all matches</option>
                    <option value="difference">Only differences</option>
                    <option value="new">Absent in {filterBaseline.datasource?.startsWith("fr3d") && !filter.datasource?.startsWith("fr3d") ? "FR3D" : "baseline"} (green)</option>
                    <option value="missing">Only in {filterBaseline.datasource?.startsWith("fr3d") && !filter.datasource?.startsWith("fr3d") ? "FR3D" : "baseline"} (red)</option>
                  </select>
                </div>


                {/if}
            </div>
          </div>
          {/if}
          
          <div class="control">
            <label class="checkbox" title="Rotate images 90° along X-axis to see the coplanarity">
              <input type="checkbox" checked={!!filter.rotX} on:change={e => filter = {...filter, rotX: e.currentTarget.checked }}>
              Rotate images
            </label>
          </div>

          <div class="control">
            <DownloadButton {filter} selectedPairType={metadata?.pair_type.join("-")} />
          </div>
        </div>

    </div>

    {:else if mode=="sql"}
      <div>
        <textarea class="textarea sql-editor" bind:value={filter.sql} style="width: 100%;"></textarea>
        <p class="help is-link">Use the SQL language to filter by anything. <code>selectedpair</code>, <code>selectedpair_f</code> and <code>selectedpair_n</code> contain the currently selected pair type, <code>_f</code> suffix are the filtered non-redundant set, <code>_n</code> suffix are the "nearly pairs". <code>selectedpair_allcontacts</code> and <code>selectedpair_allcontacts_f</code> include all basepair candidates. All other pair types are available in tables like <code>"tWW-A-A"</code> with the optional <code>_f</code> or <code>_n</code> suffix. The query runs in the browser, so run as many <code>DROP DATABASE</code>s as you please.</p>
        <p>
        </p>
      </div>
      <div style="float: right; margin-right: 2rem">
        <div class="control" style="display: inline">
          <label class="checkbox" title="Rotate images 90° along X-axis to see the coplanarity">
            <input type="checkbox" checked={!!filter.rotX} on:change={e => filter = {...filter, rotX: e.currentTarget.checked }}>
            Rotate images
          </label>
        </div>
      </div>
      {#if allowFilterBaseline}
        <div style="display: flex; align-items: center; gap: 1rem">
        {#if filter.datasource?.startsWith("allcontacts") && metadata != null}
          <button class="button" type="button" on:click={async ()=> {
            await setFr3dObservedBoundaries(true)
            if (filter.datasource == "allcontacts-boundaries-f") {
              filter = {...filter, datasource: "allcontacts-f"}
            } else if (filter.datasource == "allcontacts-boundaries") {
              filter = {...filter, datasource: "allcontacts"}
            }
          } }>
            {#if filter.datasource == "allcontacts-boundaries-f"}Edit the filter boundaries{:else}Constrain to FR3D observed ranges{/if}
          </button>
        {/if}
        {#if filterBaseline == null}
          <button class="button" type="button" on:click={() => setBaseline(structuredClone(filter), mode)}>
            Set as baseline
          </button>
          <button class="button" type="button" on:click={() => setBaseline({ ...defaultFilter(), datasource: filter.filtered ? "fr3d-f" : "fr3d" }, "gui")}>
            Compare with FR3D
          </button>
        {:else}
          <button class="button is-warning" type="button" on:click={() => setBaseline(null, mode)}>
            Exit comparison
          </button>
          <button class="button" type="button" on:click={() => setBaseline(structuredClone(filter), mode)}>
            Set current query as baseline
          </button>
          {#if filterBaseline.sql == null}
          <button class="button" type="button" on:click={() => filterBaseline = {...filterBaseline, sql: makeSqlQuery(filterBaseline, getDataSourceTable(filterBaseline)) }}>
            Edit baseline
          </button>
          {/if}
          {#if comparisonMode != null}
          <div class="select">
            <select bind:value={comparisonMode}>
              <option value="union">Show all matches</option>
              <option value="difference">Only differences</option>
              <option value="new">Absent in baseline</option>
              <option value="missing">Only in baseline</option>
            </select>
          </div>
          {/if}
          <span>Comparing with {formatFilter(filterBaseline, filter)}</span>
        {/if}
        </div>
      {/if}

      {#if filterBaseline != null && filterBaseline.sql != null}
        <div class="field">
          <label class="label is-medium" for="filter-baseline-sql-textarea">Comparing with baseline query:</label>
          <div class="control">
            <textarea class="textarea sql-editor" id="filter-baseline-sql-textarea" bind:value={filterBaseline.sql} style="width: 100%; font-color: #bbbbbb"></textarea>
          </div>
        </div>
      {/if}
    {/if}
</div>
