// To learn about scheduled functions and supported cron extensions,
// see: https://ntl.fyi/sched-func
import sgMail from "@sendgrid/mail"
import fetch from "node-fetch"

const now = Date.now();
const oneDayAgo = now - 24 * 60 * 60 * 1000;
const { NETLIFY_BLOG_SITE_ID, NETLIFY_TOKEN, SENDGRID_API_KEY } = process.env;
// export const config = {
//   // schedule: '@daily',
//   schedule: '23 11 * * *',
// }

export const handler = async function (event) {
  try {
    // Fetch the data
    const sources = await fetchNetlify(
      `ranking/sources?from=${oneDayAgo}&to=${now}&limit=100&timezone=+0000&resolution=hour`
    );
    const pages = await fetchNetlify(
      `ranking/pages?from=${oneDayAgo}&to=${now}&limit=10&timezone=+0000&resolution=hour`
    );

    // Render the email
    const html = renderEmail({ sources, pages });

    sgMail.setApiKey(SENDGRID_API_KEY);
    sgMail.setDataResidency('eu');
    await sgMail.send({
      to: process.env.EMAIL_TO,
      from: process.env.EMAIL_FROM,
      subject: "Daily Netlify Analytics Digest",
      html,
    })
    .then(() => {
      console.log('Email sent')
    })
    .catch((error) => {
      console.error(error)
    })
    
    // Response with the email's contents
    return {
      statusCode: 200,
      body: html,
    };
  } catch (e) {
    console.error(e);
    return {
      statusCode: 400,
      body: JSON.stringify({ errors: [e.toString()] }),
    };
  }
};

// Fetch data from Netlify API
function fetchNetlify(path) {
  return fetch(
    `https://analytics.services.netlify.com/v2/${NETLIFY_BLOG_SITE_ID}/${path}`,
    {
      headers: {
        Authorization: `Bearer ${NETLIFY_TOKEN}`,
      },
    }
  ).then((res) => {
    if (res.ok) {
      return res.json();
    }
    throw new Error(
      `Failed to connect to Netlify: ${res.status}: ${res.statusText}`
    );
  });
}

// Render an HTML email with the data from Netlify's Analytics API
function renderEmail({ sources, pages }) {
  const renderDate = (d) =>
    new Date(d).toLocaleString("en-GB", {
        timeZone: "Europe/London",
        timeZoneName: "short",
    })
  const styles = {
    html: "font-family: ui-sans-serif, Helvetica Neue, sans-serif;",
    ul: "margin: 0; padding: 0; list-style-type: none;",
    li: "display: flex; justify-content: space-between; align-items: center; padding: 12px 0;border-top: 1px solid #eee;",
  };

  return /*html*/`<!doctype html>
    <html style="${styles.html}">
      <head>
        <meta charset="UTF-8">
        <title>Analytics digest</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
      </head>
      <body style="max-width: 400px; margin: 0 auto; background: white; padding: 0 12px;">
        <h1>Analytics digest</h1>
        <p>
          <time>
            ${renderDate(oneDayAgo)}
          </time>
          → 
          <time>
            ${renderDate(now)}
          </time>
        </p>
        <h2>Top Sources</h2>
        <ul style="${styles.ul}">
          ${sources.data
            // Ignore "" (uknown sources and/or direct traffic)
            // as well as anything that's not double digits
            .filter(({ count, resource }) => resource && count >= 10)
            .map(
              ({ count, resource }) => /*html*/ `
              <li style="${styles.li}">
                <a style="flex-grow: 2;" href="http://${resource}">
                  ${resource}
                </a>
                <span style="font-size: 90%; margin-right: 8px; text-align: right;">${count.toLocaleString()}</span>
                <a style="font-size: 80%; text-decoration: none;" href="https://www.google.com/search?q=blog.jim-nielsen.com&as_sitesearch=${resource}&as_qdr=m">
                  🔍
                </a>
              </li>`
            )
            .join("")}
        </ul>
        <h2>Top Pages</h2>
        <ul style="${styles.ul}">
          ${pages.data
            .map(
              ({ count, resource }) => /*html*/ `
              <li style="${styles.li}">
                <a href="https://blog.jim-nielsen.com${resource}">
                  ${resource}
                </a>
                <span>${count}</span>
              </li>`
            )
            .join("")}
        </ul>
      </body>
    </html>`;
}
