import axios from "axios"
import { request } from "express";
import { response } from "express";

export const image = async(req,res) => {
  try {
    const apiURL = 'http://localhost:5000/generate_dzi';
    const url = req.body.url
    if (!url) {
        return res.status(400).json({ error: "Missing 'url' in request body" });
    }
    const response = await axios.post(apiURL, {'url':url});
    res.status(200).json(response.data);
  } catch (error) {
    console.error('Error fetching from external API:', error.message);
    res.status(500).json({ message: 'Error fetching data from external source' });
  }
}

export const init = (req,res) => {
    // frontend cahlu karna hai
    console.log("Someone Entered the server")
    res.send("<h1>Hello World</h1>")
}