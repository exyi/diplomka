<script lang="ts">
    import Pairimage from "./pairimage.svelte";
    import type { NucleotideId, PairId, PairingInfo } from "../pairing";
	import type { DetailModalViewModel } from "$lib/dbModels";

    export let pairs: PairingInfo[]
    export let rootImages: string
    export let rotImg: boolean = false
    export let imgAttachement: string
    export let videoAttachement: string
    export let videoOnHover: boolean = false
    export let onClick: (p: DetailModalViewModel) => void = () => {}

    function ntUrlIdentifier(nt: NucleotideId) {
        return `${nt.chain}_${nt.resnum}${nt.inscode??''}${nt.altloc??''}${nt.symop ? '_S' + nt.symop : ''}`
    }

    function getUrl(pair: PairId, attachement:string, opt = { rotImg }) {
        if (!pair?.nt1.pdbid || !pair?.nt2.pdbid || pair.nt1.chain == null || pair.nt2.chain == null || pair.nt1.resnum == null || pair.nt2.resnum == null)
            return undefined
        const imgName = `${pair.nt1.model > 1 ? `model${pair.nt1.model}_` : ''}${ntUrlIdentifier(pair.nt1)}-${ntUrlIdentifier(pair.nt2)}`
        return `${rootImages}/${pair.nt1.pdbid}/${imgName}${opt.rotImg ? "-rotX" : ""}${attachement}`
    }

</script>

<div class="imgcontainer">
    {#each pairs as p}
        <Pairimage pair={p} url={getUrl(p.id, imgAttachement)} videoUrl={getUrl(p.id, videoAttachement, {rotImg: false})} allowHoverVideo={videoOnHover}
            onClick={() => onClick({ pair: p, imgUrl: getUrl(p.id, imgAttachement, {rotImg: false}), rotImgUrl: getUrl(p.id, imgAttachement, {rotImg: true}), videoUrl: getUrl(p.id, videoAttachement, {rotImg: false}) }) } />
    {/each}
</div>

<style>
    .imgcontainer {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-evenly;
    }
</style>
