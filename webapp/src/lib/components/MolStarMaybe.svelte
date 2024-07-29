<script lang="ts">
    import type { PairId } from '$lib/pairing'
    import type { MolStarViewer } from './molStarBackend'
    import { onMount } from 'svelte'

    let container: HTMLDivElement
    export let pairId: PairId
    let molStarPromise: Promise<MolStarViewer>
    let molStar: MolStarViewer

    onMount(() => {
        molStarPromise = import('./molStarBackend').then(({ createMolStar }) => {
            return createMolStar(container)
        })
        molStarPromise.then((viewer) => {
            molStar = viewer
        })
    })

    $: if (pairId && molStar) {
        molStar.loadPair(pairId)
    }

</script>

<style>
    div {
        position: relative;
        width: 100%;
        height: 100%;
    }
</style>

<div bind:this={container}></div>
