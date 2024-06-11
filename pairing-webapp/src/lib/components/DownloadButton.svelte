<script lang="ts">
    import type { NucleotideFilterModel } from "$lib/dbModels";
    import { normalizePairType } from "$lib/pairing";
    import * as db from '$lib/dbInstance'

    export let filter: NucleotideFilterModel | null | undefined
    export let selectedPairType: string | null | undefined = null

    let modal: HTMLDialogElement | null = null

    function showModal(sender: HTMLElement) {
      modal.showModal()
      const pos = sender.getBoundingClientRect()
      modal.style.top = (window.scrollY+pos.bottom) + "px"
      modal.style.left = `calc(min(${window.scrollX + window.innerWidth}px - 300px, ${window.scrollX+pos.left}px))`
    }

    function dataSourceUrl(ds: NucleotideFilterModel["datasource"] | null, pairType: string | null) {
        if (!ds || !pairType) {
            return ""
        }
        pairType = normalizePairType(pairType)
        const name = ds == "allcontacts" || ds == "allcontacts-boundaries" ? `${pairType}-allcontacts` :
                     ds == "allcontacts-f" || ds == "allcontacts-boundaries-f" ? `${pairType}-filtered-allcontacts` :
                     ds == "fr3d" ? `${pairType}` :
                     ds == "fr3d-f" ? `${pairType}-filtered` :
                     ds == "fr3d-n" || ds == "fr3d-nf" ? `${pairType}-noncanonical` :
                     `${pairType}-allcontacts`;
        const file = db.parquetFiles[name]
        if (!file) return ""
        return new URL(file, db.fileBase.replace(/\/?$/, '/')).href
    }

    function successAnim(button: HTMLElement) {
        button.classList.add('is-success')
        setTimeout(() => button.classList.remove('is-success'), 1000)
    }

    function copySQL(e: HTMLElement) {
        const query = db.mainQueryHistory.history.at(-1)
        if (!query) return
        navigator.clipboard.writeText(query.sql)
        successAnim(e)
    }

    async function copyCSV(sender: HTMLElement) {
        sender.classList.add('is-loading')
        try {
            const query = db.mainQueryHistory.history.at(-1)
            if (!query) return
            const sql = `COPY (SELECT * FROM (${query.sql}) LIMIT 1000) TO 'tmp.csv' (FORMAT CSV, HEADER, sep '\\t');`
            const conn = await db.connect()
            await conn.query(sql)
            const csv = await db.getOutputFile('tmp.csv')
            const csvStr = new TextDecoder().decode(csv)
            navigator.clipboard.writeText(csvStr)
            successAnim(sender)
        }
        catch (e) {
            console.error(e)
            alert("Error copying CSV: " + e)
        }
        finally {
            sender.classList.remove('is-loading')
        }
    }

    export function magicDownloadLink(url: string, name: string) {
        const fakeAnchor = <HTMLAnchorElement> document.createElement("a");
        fakeAnchor.style.display = "none";
        document.body.appendChild(fakeAnchor);
        fakeAnchor.href = url;
        fakeAnchor.download = name
        fakeAnchor.click();
        setTimeout(() => fakeAnchor.remove(), 1000);
    }


    async function exportFile(sender: HTMLElement, fmt: "csv" | "parquet") {
        sender.classList.add('is-loading')
        try {
            const query = db.mainQueryHistory.history.at(-1)
            if (!query) return
            const sql = `COPY (${query.sql}) TO 'tmp.${fmt}' (FORMAT ${fmt}, ${fmt == "csv" ? "header" : "compression zstd"});`
            const conn = await db.connect()
            await conn.query(sql)
            const file = await db.getOutputFile('tmp.' + fmt)
            const link = URL.createObjectURL(new Blob([file], {type: fmt == "csv" ? "text/csv" : "application/octet-stream"}))
            magicDownloadLink(link, `basepairs-export-${selectedPairType}.${fmt}`)
            setTimeout(() => URL.revokeObjectURL(link), 60_000)
            successAnim(sender)
        }
        catch (e) {
            console.error(e)
            alert("Error copying CSV: " + e)
        }
        finally {
            sender.classList.remove('is-loading')
        }
    }
</script>

<style>
    dialog .button {
        min-width: 100%;
        transition: background-color 0.5s linear;
    }
</style>

<div>
    <button class="button" on:click={ev => showModal(ev.currentTarget)}>Download ⏷ </button>
    <dialog bind:this={modal} style="position: absolute;margin: 0;max-width: 300px" on:click={ev => {
        if (ev.target != modal) {
            return
        }
        const rect = modal.getBoundingClientRect();
        const isInDialog = (rect.top <= ev.clientY && ev.clientY <= rect.top + rect.height &&
          rect.left <= ev.clientX && ev.clientX <= rect.left + rect.width);
        if (!isInDialog) {
          modal.close();
        }
      }}>
      <div>
        <a class="button" download href={dataSourceUrl(filter?.datasource, selectedPairType)}>
            Data source (Parquet)
        </a>
      </div>
      <div>
      <button class="button" on:click={ev => exportFile(ev.currentTarget, "parquet")}>
        Results (Parquet)
      </button>
      </div>
      <div>
      <button class="button" on:click={ev => exportFile(ev.currentTarget, "csv")}>
        Results (CSV)
      </button>
      </div>
      <div>
      <button class="button" on:click={ev => copySQL(ev.currentTarget)}>
        ⎘ Copy query SQL
      </button>
      </div>
      <div>
      <button class="button" on:click={ev => copyCSV(ev.currentTarget)}>
        ⎘ Copy results TSV
      </button>
      </div>
    </dialog>
</div>
