"""XMP sidecars: standard face regions (MWG) + the WikiTree ids as keywords, readable by Lightroom, digiKam, Apple Photos, etc."""
from __future__ import annotations

from xml.sax.saxutils import quoteattr, escape


def build_xmp(width: int, height: int, regions: list[dict], subjects: list[str], description: str = "") -> str:
    """regions: [{'name': str, 'x','y','w','h'}] with top-left, normalized 0-1 coordinates."""
    items = "".join(
        f'<rdf:li><rdf:Description mwg-rs:Name={quoteattr(r["name"])} mwg-rs:Type="Face">'
        f'<mwg-rs:Area stArea:x="{r["x"] + r["w"] / 2:.5f}" stArea:y="{r["y"] + r["h"] / 2:.5f}" '
        f'stArea:w="{r["w"]:.5f}" stArea:h="{r["h"]:.5f}" stArea:unit="normalized"/></rdf:Description></rdf:li>'
        for r in regions)
    keywords = "".join(f"<rdf:li>{escape(s)}</rdf:li>" for s in subjects)
    names = "".join(f"<rdf:li>{escape(r['name'])}</rdf:li>" for r in regions)
    desc = (f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{escape(description)}</rdf:li></rdf:Alt></dc:description>'
            if description else "")
    return f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
    xmp:CreatorTool="WikiTree PhotoTag"
    xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
    xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#"
    xmlns:stDim="http://ns.adobe.com/xap/1.0/sType/Dimensions#">
   {desc}
   <dc:subject><rdf:Bag>{keywords}</rdf:Bag></dc:subject>
   <Iptc4xmpExt:PersonInImage><rdf:Bag>{names}</rdf:Bag></Iptc4xmpExt:PersonInImage>
   <mwg-rs:Regions rdf:parseType="Resource">
    <mwg-rs:AppliedToDimensions stDim:w="{width}" stDim:h="{height}" stDim:unit="pixel"/>
    <mwg-rs:RegionList><rdf:Bag>{items}</rdf:Bag></mwg-rs:RegionList>
   </mwg-rs:Regions>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""
