import React, { useState } from "react";
import axios from "axios";

export default function App(){
  const [brief, setBrief] = useState({product:"SparkX Energy Drink", audience:"students", tone:"energetic", goal:"awareness", platform:["instagram"]});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setLoading(true);
    try{
      const resp = await axios.post("/generate", brief);
      setResult(resp.data);
    }catch(e){
      alert("Error: " + e.message);
    }finally{
      setLoading(false);
    }
  }

  return (<div style={{padding:20,fontFamily:"Arial"}}>
    <h2>Campaign Generator Demo</h2>
    <div>
      <label>Product: <input value={brief.product} onChange={e=>setBrief({...brief,product:e.target.value})} /></label>
    </div>
    <div style={{marginTop:10}}>
      <button onClick={submit} disabled={loading}>{loading? "Generating...":"Generate"}</button>
    </div>
    <div style={{marginTop:20}}>
      {result && result.top_assets && result.top_assets.map((a,idx)=>(
        <div key={idx} style={{border:"1px solid #ddd",padding:10,marginBottom:10}}>
          <div><strong>Score:</strong> {a.score.toFixed(3)}</div>
          <div><strong>Copy:</strong> {a.copy}</div>
          <div><img src={a.image_url} style={{maxWidth:300}} alt="asset" /></div>
        </div>
      ))}
    </div>
  </div>)
}
