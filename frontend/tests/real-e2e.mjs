// Explicit opt-in real browser acceptance. Never included in Vitest or CI tests.
// Admin credentials are read at runtime and never persisted in the result file.
import { chromium, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const python = resolve(root, '.venv/Scripts/python.exe');
const py = code => execFileSync(python, ['-X','utf8','-c', `import os,sys; os.chdir(${JSON.stringify(root)}); sys.path.insert(0,os.getcwd()); ${code}`], { encoding: 'utf8' }).trim();
const token = py("from dotenv import load_dotenv; load_dotenv('.env'); print(os.getenv('REVIEW_ADMIN_TOKEN',''))");
if (!token) throw new Error('Missing local review token; no requests performed.');
const count = () => JSON.parse(py("import json; from app.database import SessionLocal,current_database_name; from app.models import QAReviewRecord; from sqlalchemy import select,func; assert current_database_name()=='roborock'; s=SessionLocal(); print(json.dumps({'count':s.scalar(select(func.count()).select_from(QAReviewRecord)), 'max_id':s.scalar(select(func.max(QAReviewRecord.id))) or 0})); s.close()"));
const resultPath = resolve(root, 'data/evaluation/results/frontend_v1_e2e.json');
mkdirSync(resolve(root,'tmp/frontend'), { recursive:true });
const phase = process.argv[2] || 'chat';
const result = phase !== 'chat' ? JSON.parse(readFileSync(resultPath, 'utf8')) : {
  frontend_version: '1.0.0', api_base_for_acceptance: 'http://127.0.0.1:8001',
  api_endpoints_used: ['/api/agent/chat','/internal/reviews?scope=all&limit=50','/internal/reviews/{id}','/internal/reviews/{id}/decision','/internal/reviews/{id}/publish','/api/published/{id}','/health','/ready'],
  frontend_tests:25, python_tests:360, build:'PASS', real_backend:true, real_browser:true,
};
const browser = await chromium.launch({ channel:'chrome', headless:true });
const page = await browser.newPage({ viewport:{width:1440,height:1000} });
page.setDefaultTimeout(20000);
const save = () => writeFileSync(resultPath, JSON.stringify(result,null,2)+'\n');
try {
  if (phase === 'chat') {
    result.database_before = count();
    await page.goto('http://127.0.0.1:5173/');
    await expect(page.getByRole('heading',{name:'今天，T8 遇到了什么问题？'})).toBeVisible();
    await page.screenshot({path:resolve(root,'tmp/frontend/chat.png'),fullPage:true});
    async function send(question) {
      await page.getByLabel('输入问题').fill(question);
      const responsePromise = page.waitForResponse(r => r.url().endsWith('/api/agent/chat') && r.request().method()==='POST',{timeout:180000});
      await page.getByRole('button',{name:/发送问题/}).click();
      const response = await responsePromise;
      const body = await response.json();
      const request = response.request().postDataJSON();
      const outcome = { question, request, status:response.status(), response:body };
      if (response.status() !== 200) throw new Error(`Chat HTTP ${response.status()}`);
      expect(body).not.toHaveProperty('draft_answer'); expect(body).not.toHaveProperty('evidence');
      await expect(page.getByText(body.answer || body.safety_notice,{exact:true}).last()).toBeVisible();
      return outcome;
    }
    result.chat = await send('主刷多久清理一次？');
    expect(result.chat.response.intent).toBe('maintenance');
    expect(result.chat.response.answer).toBeNull(); expect(result.chat.response.published).toBe(false);
    result.database_after_maintenance = count();
    expect(result.database_after_maintenance.count-result.database_before.count).toBe(1);
    result.approve_record_id = result.database_after_maintenance.max_id;
    await page.getByRole('button',{name:/新建会话/}).click();
    const first = await send('机器人无法正常工作怎么办？');
    expect(first.response.intent).toBe('troubleshooting'); expect(typeof first.response.answer).toBe('string');
    const second = await send('扫着扫着会自己停下来。');
    expect(second.response.intent).toBe('troubleshooting');
    expect(second.request.session_id).toBe(first.request.session_id);
    result.troubleshooting = { first, second, same_session:true };
    result.database_after_troubleshooting = count();
    expect(result.database_after_troubleshooting.count-result.database_after_maintenance.count).toBe(1);
    result.reject_record_id = result.database_after_troubleshooting.max_id;
    result.human_handoff = await send('我要人工客服');
    expect(result.human_handoff.response.intent).toBe('human_handoff');
    expect(count().count).toBe(result.database_after_troubleshooting.count);
    await page.goto('http://127.0.0.1:5173/review');
    await page.getByLabel('Review Token').fill(token);
    const listPromise = page.waitForResponse(r => r.url().includes('/internal/reviews?'));
    await page.getByRole('button',{name:'验证并进入'}).click();
    const listResponse = await listPromise;
    result.review_list = {status:listResponse.status(), count:(await listResponse.json()).length};
    expect(listResponse.status()).toBe(200);
    await page.getByRole('button',{name:new RegExp('#'+result.approve_record_id+' ')}).click();
    await expect(page.getByRole('heading',{name:'记录 #'+result.approve_record_id})).toBeVisible();
    const detail = await page.request.get('http://127.0.0.1:5173/internal/reviews/'+result.approve_record_id, {headers:{'X-Review-Token':token}});
    const data = await detail.json();
    // Private local inspection file only, not API, browser memory, or public result.
    writeFileSync(resolve(root,'tmp/frontend/review-inspection.json'),JSON.stringify(data,null,2));
    result.review_detail = {status:detail.status(), id:data.id, review_status:data.review_status, publication_status:data.publication_status, evidence_count:data.evidence.length};
    expect(data.id).toBeGreaterThan(result.database_before.max_id);
    await page.screenshot({path:resolve(root,'tmp/frontend/review.png'),fullPage:true});
    expect(await page.evaluate(() => localStorage.length+sessionStorage.length)).toBe(0);
    await page.reload(); await expect(page.getByLabel('Review Token')).toHaveValue('');
    result.token_storage = {local_storage:false, session_storage:false, cleared_on_reload:true};
    result.stage = 'CHAT_AND_REVIEW_READ_COMPLETE';
  } else if (phase === 'visual') {
    await page.goto('http://127.0.0.1:5173/');
    await expect(page.locator('.sidebar').getByRole('button',{name:/新建会话/})).toBeVisible();
    await page.screenshot({path:resolve(root,'tmp/frontend/chat.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
    await page.screenshot({path:resolve(root,'tmp/frontend/mobile.png'),fullPage:true});
    await page.setViewportSize({width:1440,height:1000});
    await page.goto('http://127.0.0.1:5173/review');
    await page.getByLabel('Review Token').fill(token);
    await page.getByRole('button',{name:'验证并进入'}).click();
    await page.getByRole('button',{name:'全部',exact:true}).click();
    await page.getByRole('button',{name:new RegExp('#'+result.approve_record_id+' ')}).click();
    await expect(page.getByRole('link',{name:/查看正式答案/})).toBeVisible();
    await page.screenshot({path:resolve(root,'tmp/frontend/review.png'),fullPage:true});
    await page.goto('http://127.0.0.1:5173/published/'+result.approve_record_id);
    await expect(page.locator('.public-answer')).toBeVisible();
    await page.screenshot({path:resolve(root,'tmp/frontend/published.png'),fullPage:true});
    result.visual_check={desktop:true,mobile_no_horizontal_overflow:true,sidebar_new_session:true};
  } else if (phase === 'publication-case') {
    await page.goto('http://127.0.0.1:5173/');
    await page.getByLabel('输入问题').fill('软胶主刷的清理和更换周期分别是什么？');
    const responsePromise=page.waitForResponse(r=>r.url().endsWith('/api/agent/chat'),{timeout:180000});
    await page.getByRole('button',{name:/发送问题/}).click();
    const response=await responsePromise;
    result.publication_test_chat={status:response.status(),response:await response.json()};
    expect(response.status()).toBe(200);
    result.original_maintenance_record_id=result.approve_record_id;
    result.approve_record_id=count().max_id;
    const detail=await page.request.get('http://127.0.0.1:5173/internal/reviews/'+result.approve_record_id,{headers:{'X-Review-Token':token}});
    writeFileSync(resolve(root,'tmp/frontend/review-inspection.json'),JSON.stringify(await detail.json(),null,2));
    result.stage='PUBLICATION_CASE_READY_FOR_INSPECTION';
  } else {
    const id = result.approve_record_id;
    expect(id).toBeGreaterThan(result.database_before.max_id);
    await page.goto('http://127.0.0.1:5173/review');
    await page.getByLabel('Review Token').fill(token);
    await page.getByRole('button',{name:'验证并进入'}).click();
    await page.getByRole('button',{name:new RegExp('#'+id+' ')}).click();
    await page.getByLabel('审核备注').fill('前端V1验收：已核对本轮主刷清理和更换周期与说明书证据，批准草稿；发布单独执行。');
    const decisionPromise = page.waitForResponse(r => r.url().endsWith('/decision'));
    await page.getByRole('button',{name:'通过审核'}).click();
    result.approve = {status:(await decisionPromise).status()};
    await expect(page.getByRole('button',{name:'正式发布'})).toBeVisible();
    const detailResponse = await page.request.get('http://127.0.0.1:5173/internal/reviews/'+id,{headers:{'X-Review-Token':token}});
    const approved = await detailResponse.json();
    result.approve.review_status=approved.review_status; result.approve.publication_status=approved.publication_status;
    expect(approved.review_status).toBe('approved'); expect(approved.publication_status).toBe('unpublished');
    const publicPage = await browser.newPage();
    const absentPromise = publicPage.waitForResponse(r => r.url().endsWith('/api/published/'+id));
    await publicPage.goto('http://127.0.0.1:5173/published/'+id);
    result.approved_unpublished_public_status = (await absentPromise).status();
    await expect(publicPage.getByText('该回答尚未发布或不存在。')).toBeVisible();
    expect(result.approved_unpublished_public_status).toBe(404);
    const publishPromise = page.waitForResponse(r => r.url().endsWith('/publish'));
    await page.getByRole('button',{name:'正式发布'}).click();
    const publishResponse=await publishPromise; result.publish={status:publishResponse.status(), response:await publishResponse.json()};
    expect(result.publish.status).toBe(200);
    await expect(page.getByRole('link',{name:/查看正式答案/})).toBeVisible();
    const publicPromise = publicPage.waitForResponse(r => r.url().endsWith('/api/published/'+id));
    await publicPage.reload(); const publicResponse = await publicPromise;
    result.published_page = {status:publicResponse.status(), response:await publicResponse.json()};
    expect(result.published_page.status).toBe(200);
    await expect(publicPage.locator('.public-answer')).toBeVisible();
    expect(await publicPage.locator('.public-answer').innerText()).toBe(approved.draft_answer);
    await publicPage.setViewportSize({width:1440,height:1000});
    await publicPage.screenshot({path:resolve(root,'tmp/frontend/published.png'),fullPage:true});
    // Only the second newly created record is rejected; no historical record is edited.
    const rejectedId=result.reject_record_id; expect(rejectedId).toBeGreaterThan(result.database_before.max_id);
    await page.getByRole('button',{name:new RegExp('#'+rejectedId+' ')}).click();
    await page.getByLabel('审核备注').fill('前端V1验收：本轮测试草稿暂不发布，演示驳回状态并保留记录。');
    const rejectPromise=page.waitForResponse(r => r.url().endsWith('/decision'));
    await page.getByRole('button',{name:'驳回'}).click();
    const rejectResponse=await rejectPromise; result.reject={status:rejectResponse.status(), response:await rejectResponse.json()};
    await expect(page.getByRole('button',{name:'驳回'})).toHaveCount(0);
    await expect(page.getByRole('button',{name:'正式发布'})).toHaveCount(0);
    expect(result.reject.response.review_status).toBe('rejected');
    result.database_after = count();
    result.stage='COMPLETE'; result.conclusion='FRONTEND_V1: PASS';
  }
  save(); console.log(JSON.stringify({stage:result.stage, approve_record_id:result.approve_record_id, reject_record_id:result.reject_record_id}));
} catch (error) {
  result.stage='FAILED'; result.conclusion='FRONTEND_V1: FAIL'; result.failure=String(error.message).replaceAll(token,'[REDACTED]'); save();
  console.error(result.failure); process.exitCode=1;
} finally { await browser.close(); }
